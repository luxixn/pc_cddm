"""
PC-CDDM 训练主入口。

职责:
    1. 读取 yaml 配置, 支持点式 override (Notebook 注入路径用)
    2. 构造数据/模型/优化器/调度器
    3. 运行训练循环: AMP 混合精度 + 梯度裁剪 + 梯度累积
    4. 每 N epoch 跑一次 DDIM 快速验证, 算 NMSE 跟踪 best
    5. 保存 ckpt (latest / best / 带 epoch 编号), 自动清理旧的
    6. 会话感知退出: wallclock 接近上限时保存 latest 并优雅 break
    7. 断点续训: resume_from=null/auto/path 三态

使用:
    命令行:  python -m pc_cddm.train --config configs/default.yaml
                                     [--override paths.data=/kaggle/input/...]
    Notebook: from pc_cddm.train import main; main("configs/default.yaml")

红线:
    训练 loss (噪声 MSE + PSD MSE) 不能作为模型质量指标。
    模型质量必须由完整反向采样后的 NMSE 决定 -> validate() 走 DDIM 全链。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pc_cddm.data import IQDataset
from pc_cddm.diffusion import (
    DiffusionSchedule,
    compute_pcddm_loss,
    sample,
)
from pc_cddm.models.unet1d import UNet1D
from pc_cddm.models.condition_encoder import ConditionEncoder
from pc_cddm.utils.psd import welch_psd_log
from pc_cddm.utils.metrics import compute_eval_metrics
from pc_cddm.utils.logging import (
    Logger,
    CheckpointManager,
    WallclockTimer,
    setup_run_dir,
)


# ============================================================================
# 配置加载 (统一处理 Windows 中文 yaml 编码 + 点式 override)
# ============================================================================
def load_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    加载 yaml 配置, 支持点式 override 与 paths.data -> data.h5_path 注入。

    Args:
        config_path: yaml 文件路径
        overrides:   点式 key 字典, e.g. {"paths.data": "/kaggle/...", "train.lr": 1e-4}

    Returns:
        完整 cfg dict, 已注入 data.h5_path
    """
    # 关键: encoding='utf-8' 防 Windows gbk 解码失败
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 点式 override
    if overrides:
        for key, value in overrides.items():
            _set_nested(cfg, key, value)

    # paths.data -> data.h5_path 自动注入
    cfg["data"]["h5_path"] = cfg["paths"]["data"]

    return cfg


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """在嵌套字典里按点式 key 设值, e.g. _set_nested(d, 'a.b.c', 1)。"""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


# ============================================================================
# 随机种子
# ============================================================================
def set_seed(seed: int) -> None:
    """统一种子。注意 DataLoader 多 worker 仍需 worker_init_fn 才完全可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 训练单步: 一次前向 + loss
# ============================================================================
def train_step(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    unet: nn.Module,
    condition_encoder: nn.Module,
    schedule: DiffusionSchedule,
    cfg: dict[str, Any],
    device: torch.device,
) -> "_StepResult":
    """
    一个 mini-batch 的训练前向, 返回 LossOutput-like 结果与中间张量供上层 backward。

    步骤:
        1. (y, x_0, snr) -> n = y - x_0
        2. psd_gt_log = welch_psd_log(n)
        3. 采 t ~ U[0, T-1], eps ~ N(0, I)
        4. x_t = q_sample(x_0, t, eps)
        5. c = condition_encoder(t, snr, psd_gt_log)
        6. eps_pred = unet(x_t, c)
        7. loss = compute_pcddm_loss(...)

    Args:
        batch:   (y, x_0, snr_db) 来自 DataLoader, 已 device-移动 / 不移动均可,
                 内部统一 .to(device)
        unet, condition_encoder, schedule: 模型组件
        cfg:     完整配置
        device:  目标设备
    """
    y, x_0, snr_db = batch
    y     = y.to(device, non_blocking=True)      # [B, 2, L]
    x_0   = x_0.to(device, non_blocking=True)    # [B, 2, L]
    snr_db = snr_db.to(device, non_blocking=True)  # [B]

    B = y.size(0)
    T = schedule.T

    # 1) 真实噪声及其 PSD (loss 与条件输入共用一份)
    n = y - x_0                                    # [B, 2, L]
    psd_gt_log = welch_psd_log(
        n,
        nperseg=cfg["psd"]["nperseg"],
        noverlap=cfg["psd"]["noverlap"],
        fs=cfg["psd"]["fs"],
        eps=cfg["psd"]["eps"],
        detrend=cfg["psd"]["detrend"],
    )                                              # [B, 256]

    # 2) 采 t & eps
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)  # [B]
    eps = torch.randn_like(x_0)                    # [B, 2, L]

    # 3) 加噪
    x_t = schedule.q_sample(x_0, t, eps)           # [B, 2, L]

    # 4) 条件编码
    c = condition_encoder(t, snr_db, psd_gt_log)   # [B, cond_dim]

    # 5) 网络前向
    eps_pred = unet(x_t, c)                        # [B, 2, L]

    # 6) 损失 (PSD threshold = T * ratio)
    psd_threshold = int(T * cfg["diffusion"]["psd_loss_threshold_ratio"])
    out = compute_pcddm_loss(
        eps_pred=eps_pred,
        eps_target=eps,
        x_t=x_t,
        y=y,
        t=t,
        schedule=schedule,
        psd_gt_log=psd_gt_log,
        lambda_psd=cfg["train"]["lambda_psd"],
        psd_loss_threshold=psd_threshold,
        psd_nperseg=cfg["psd"]["nperseg"],
        psd_noverlap=cfg["psd"]["noverlap"],
        psd_fs=cfg["psd"]["fs"],
        psd_eps=cfg["psd"]["eps"],
        psd_detrend=cfg["psd"]["detrend"],
    )

    return _StepResult(loss=out.total, diff=out.diff, psd=out.psd, n_psd=out.n_psd_samples)


# 简单 dataclass 替代, 不用引入 dataclasses 模块
class _StepResult:
    __slots__ = ("loss", "diff", "psd", "n_psd")

    def __init__(self, loss, diff, psd, n_psd):
        self.loss = loss
        self.diff = diff
        self.psd = psd
        self.n_psd = n_psd


# ============================================================================
# 验证循环 (DDIM 快速反向采样, 算 NMSE)
# ============================================================================
@torch.no_grad()
def validate(
    val_loader: DataLoader,
    unet: nn.Module,
    condition_encoder: nn.Module,
    schedule: DiffusionSchedule,
    cfg: dict[str, Any],
    device: torch.device,
    logger: Logger,
) -> dict[str, float]:
    """
    跑完整反向采样得 x̂_0, 算 NMSE / OutSNR。

    红线: 用 DDIM 加速版做训练时验证 (val_ddim_steps 步), 不用训练 loss 替代。

    Returns:
        dict[str, float]: 含 nmse, out_snr_db, in_snr_db, snr_gain_db
    """
    unet.eval()
    condition_encoder.eval()

    nmse_per_all: list[torch.Tensor] = []
    out_db_per_all: list[torch.Tensor] = []
    in_db_per_all: list[torch.Tensor] = []

    for batch in val_loader:
        y, x_0, snr_db = batch
        y     = y.to(device, non_blocking=True)
        x_0   = x_0.to(device, non_blocking=True)
        snr_db = snr_db.to(device, non_blocking=True)

        x_hat = sample(
            unet, condition_encoder, schedule,
            y=y, snr_db=snr_db,
            method="ddim",
            num_inference_steps=cfg["train"]["val_ddim_steps"],
            psd_refine_interval=cfg["eval"]["psd_refine_interval"],
            psd_nperseg=cfg["psd"]["nperseg"],
            psd_noverlap=cfg["psd"]["noverlap"],
            psd_eps=cfg["psd"]["eps"],
            progress=False,
        )  # [B, 2, L]

        m = compute_eval_metrics(x_hat, x_0, y=y)
        nmse_per_all.append(m["nmse_per_sample"])
        out_db_per_all.append(m["out_snr_db_per_sample"])
        in_db_per_all.append(m["in_snr_db_per_sample"])

    nmse_all   = torch.cat(nmse_per_all)
    out_db_all = torch.cat(out_db_per_all)
    in_db_all  = torch.cat(in_db_per_all)

    metrics = {
        "nmse":         float(nmse_all.mean().item()),
        "out_snr_db":   float(out_db_all.mean().item()),
        "in_snr_db":    float(in_db_all.mean().item()),
        "snr_gain_db":  float((out_db_all - in_db_all).mean().item()),
        "n_samples":    int(nmse_all.numel()),
    }
    logger.info(
        f"[val] N={metrics['n_samples']}  NMSE={metrics['nmse']:.4f}  "
        f"OutSNR={metrics['out_snr_db']:.2f}dB  "
        f"InSNR={metrics['in_snr_db']:.2f}dB  "
        f"Gain={metrics['snr_gain_db']:.2f}dB"
    )

    return metrics


# ============================================================================
# 续训状态恢复
# ============================================================================
def maybe_resume(
    cfg: dict[str, Any],
    ckpt_mgr: CheckpointManager,
    unet: nn.Module,
    condition_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    logger: Logger,
) -> tuple[int, int, float]:
    """
    根据 paths.resume_from 决定是否恢复:
        null   -> 从头训
        "auto" -> 自动找 ckpt_latest.pt, 找不到从头训
        path   -> 强制从该路径加载

    Returns:
        (start_epoch, global_step, best_val_nmse)
    """
    rf = cfg["paths"]["resume_from"]
    if rf is None:
        logger.info("[resume] resume_from=null, 从头训")
        return 1, 0, float("inf")

    if rf == "auto":
        ckpt_path = ckpt_mgr.find_latest()
        if ckpt_path is None:
            logger.info("[resume] auto 模式但未找到任何 ckpt, 从头训")
            return 1, 0, float("inf")
    else:
        ckpt_path = Path(rf)

    logger.info(f"[resume] 从 {ckpt_path} 加载")
    state = ckpt_mgr.load(ckpt_path, map_location=device)

    unet.load_state_dict(state["unet_state"])
    condition_encoder.load_state_dict(state["ce_state"])
    optimizer.load_state_dict(state["optim_state"])
    if scaler is not None and state.get("scaler_state") is not None:
        scaler.load_state_dict(state["scaler_state"])

    start_epoch = int(state["epoch"]) + 1   # 从下一个 epoch 开始
    global_step = int(state.get("global_step", 0))
    best_val_nmse = float(state.get("best_val_nmse", float("inf")))
    logger.info(
        f"[resume] 恢复完成: start_epoch={start_epoch}, "
        f"global_step={global_step}, best_val_nmse={best_val_nmse:.4f}"
    )
    return start_epoch, global_step, best_val_nmse


# ============================================================================
# 主训练函数
# ============================================================================
def main(
    config_path: str | Path = "configs/default.yaml",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    主训练入口。返回训练结束时的状态摘要 (供 Notebook 检查)。
    """
    # ---- 0. 配置 ----
    cfg = load_config(config_path, overrides=overrides)
    set_seed(cfg["misc"]["seed"])

    device_str = cfg["misc"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # ---- 1. 目录 / 日志 ----
    paths = setup_run_dir(cfg["paths"]["output_root"], cfg["paths"]["exp_name"])
    logger = Logger(paths["log_file"])
    logger.info(f"=== PC-CDDM 训练 ===  device={device}")

    # 配置存档 (便于追溯)
    with open(paths["config_file"], "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    logger.info(f"[init] 配置已存档到 {paths['config_file']}")

    # ---- 2. 数据 ----
    train_ds = IQDataset.from_config(cfg["data"], split="train")
    val_ds   = IQDataset.from_config(cfg["data"], split="val")
    logger.info(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ---- 3. 模型 / 调度 / 优化器 ----
    unet = UNet1D.from_config(cfg["model"]).to(device)
    condition_encoder = ConditionEncoder.from_config(cfg["model"], cfg["psd"]).to(device)
    schedule = DiffusionSchedule.from_config(cfg["diffusion"]).to(device)

    n_params = (
        sum(p.numel() for p in unet.parameters())
        + sum(p.numel() for p in condition_encoder.parameters())
    )
    logger.info(f"[model] 总参数 {n_params:,}")

    # 优化器统一覆盖 unet + ce
    params = list(unet.parameters()) + list(condition_encoder.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # AMP scaler (CPU 关闭)
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(f"[amp] enabled={use_amp}")

    # ---- 4. ckpt 管理器 + 续训 ----
    ckpt_mgr = CheckpointManager(paths["ckpt_dir"], keep_last_n=cfg["train"]["keep_last_n_ckpts"])
    start_epoch, global_step, best_val_nmse = maybe_resume(
        cfg, ckpt_mgr, unet, condition_encoder, optimizer, scaler, device, logger,
    )

    # ---- 5. wallclock 计时 ----
    timer = WallclockTimer(max_hours=cfg["train"]["max_wallclock_hours"])
    early_stop_reason = None

    # ---- 6. 训练主循环 ----
    log_every       = cfg["train"]["log_every"]
    grad_clip       = cfg["train"]["grad_clip"]
    grad_accum      = cfg["train"]["grad_accum_steps"]
    val_every       = cfg["train"]["val_every_epochs"]
    save_every      = cfg["train"]["save_every_epochs"]
    num_epochs      = cfg["train"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs + 1):
        unet.train()
        condition_encoder.train()
        epoch_loss_sum = 0.0
        epoch_diff_sum = 0.0
        epoch_psd_sum  = 0.0
        epoch_steps    = 0
        epoch_t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for batch_i, batch in enumerate(train_loader):
            global_step += 1
            accum_counter += 1

            # 前向 (AMP)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    res = train_step(batch, unet, condition_encoder, schedule, cfg, device)
                    loss_for_backward = res.loss / grad_accum
                scaler.scale(loss_for_backward).backward()
            else:
                res = train_step(batch, unet, condition_encoder, schedule, cfg, device)
                loss_for_backward = res.loss / grad_accum
                loss_for_backward.backward()

            # 梯度累积满 -> step
            if accum_counter == grad_accum:
                if use_amp:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            # 日志
            epoch_loss_sum += res.loss.item()
            epoch_diff_sum += res.diff.item()
            epoch_psd_sum  += res.psd.item()
            epoch_steps    += 1

            if global_step % log_every == 0:
                logger.info(
                    f"[train] ep={epoch} step={global_step} "
                    f"L={res.loss.item():.4f} "
                    f"L_diff={res.diff.item():.4f} L_psd={res.psd.item():.4f} "
                    f"n_psd={res.n_psd}/{batch[0].size(0)}"
                )

        # epoch 末汇总
        epoch_dt = time.time() - epoch_t0
        logger.info(
            f"[epoch {epoch}] avg L={epoch_loss_sum/epoch_steps:.4f}  "
            f"L_diff={epoch_diff_sum/epoch_steps:.4f}  "
            f"L_psd={epoch_psd_sum/epoch_steps:.4f}  "
            f"耗时 {epoch_dt:.1f}s  累计 {timer.elapsed_hours():.2f}h"
        )

        # ---- 验证 ----
        is_best = False
        if epoch % val_every == 0:
            val_metrics = validate(
                val_loader, unet, condition_encoder, schedule, cfg, device, logger,
            )
            if val_metrics["nmse"] < best_val_nmse:
                best_val_nmse = val_metrics["nmse"]
                is_best = True
                logger.info(f"[best] new best NMSE {best_val_nmse:.4f}")

        # ---- 保存 ckpt ----
        if epoch % save_every == 0 or is_best:
            state = {
                "epoch":         epoch,
                "global_step":   global_step,
                "best_val_nmse": best_val_nmse,
                "unet_state":    unet.state_dict(),
                "ce_state":      condition_encoder.state_dict(),
                "optim_state":   optimizer.state_dict(),
                "scaler_state":  scaler.state_dict() if scaler is not None else None,
                "config":        cfg,
            }
            ckpt_path = ckpt_mgr.save(state, epoch=epoch, is_best=is_best)
            logger.info(f"[ckpt] saved {ckpt_path.name}{' (best)' if is_best else ''}")

        # ---- 会话感知退出 ----
        if timer.exceeded():
            early_stop_reason = "wallclock_exceeded"
            logger.info(
                f"[wallclock] 已用 {timer.elapsed_hours():.2f}h, "
                f"超 {cfg['train']['max_wallclock_hours']}h 上限, 提前退出"
            )
            # 保险再存一次 latest (不动 best)
            state = {
                "epoch":         epoch,
                "global_step":   global_step,
                "best_val_nmse": best_val_nmse,
                "unet_state":    unet.state_dict(),
                "ce_state":      condition_encoder.state_dict(),
                "optim_state":   optimizer.state_dict(),
                "scaler_state":  scaler.state_dict() if scaler is not None else None,
                "config":        cfg,
            }
            ckpt_mgr.save(state, epoch=epoch, is_best=False)
            break

    logger.info(f"=== 训练结束 ===  reason={early_stop_reason or 'all_epochs_done'}  "
                f"best_val_nmse={best_val_nmse:.4f}")
    logger.close()

    return {
        "epochs_completed":   epoch,
        "global_step":        global_step,
        "best_val_nmse":      best_val_nmse,
        "early_stop_reason":  early_stop_reason,
        "run_dir":            str(paths["run_dir"]),
    }


# ============================================================================
# 命令行
# ============================================================================
def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """
    将 CLI --override key=value 列表解析成 dict。

    值类型推断顺序:
        1. yaml.safe_load: 处理 true/false/null/list/正常数字
        2. 若结果仍是 str, 尝试 int/float 兜底
           (修复 YAML 1.1 下 "1e-3" 不被识别为数字的坑)
        3. 否则保留为字符串
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--override 必须是 key=value 形式, 实际 '{item}'")
        k, v = item.split("=", 1)
        parsed = yaml.safe_load(v)
        if isinstance(parsed, str):
            # 兜底: 像 "1e-3" / "2e+5" 这种 YAML 不识别的科学记数法
            try:
                parsed = int(parsed)
            except ValueError:
                try:
                    parsed = float(parsed)
                except ValueError:
                    pass  # 保留为字符串
        out[k.strip()] = parsed
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PC-CDDM 训练")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[],
                        help="点式覆盖, 可多次, e.g. --override paths.data=/x.h5 --override train.lr=1e-4")
    args = parser.parse_args()

    overrides = _parse_overrides(args.override) if args.override else None
    main(args.config, overrides=overrides)
