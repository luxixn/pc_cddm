"""
PC-CDDM 评估主入口 (论文最终结果生成)。

职责:
    1. 读取 yaml + 指定 ckpt (默认 ckpt_best.pt)
    2. 构造数据集 (默认 val 分片)
    3. 加载模型权重
    4. 走完整 DDPM 反向采样 (1000 步 + PSD 重估每 50 步) 得 x̂_0
    5. 算 NMSE / OutSNR / SNR Gain, 按输入 SNR 档位分组
    6. 输出 summary yaml + 可选 per-sample CSV

红线:
    论文表格必须用 DDPM 完整反向链 (use_ddim=False, full_sampling_steps=T)。
    DDIM 仅供 dev 时快速冒烟, 不进论文。

使用:
    命令行: python -m pc_cddm.evaluate --config configs/default.yaml
                                       --ckpt runs/exp_default/ckpts/ckpt_best.pt
                                       [--split val|all]
                                       [--use_ddim]              # dev only
                                       [--max_samples 100]       # dev only
                                       [--save_per_sample]       # 输出 CSV
                                       [--eval_name my_eval]     # 输出子目录名
    Notebook: from pc_cddm.evaluate import main; main(...)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from pc_cddm.data import IQDataset
from pc_cddm.diffusion import DiffusionSchedule, sample
from pc_cddm.models.unet1d import UNet1D
from pc_cddm.models.condition_encoder import ConditionEncoder
from pc_cddm.utils.metrics import compute_eval_metrics, group_by_snr
from pc_cddm.utils.logging import Logger, CheckpointManager, setup_run_dir
from pc_cddm.train import load_config, set_seed   # 复用


# ============================================================================
# 评估主循环
# ============================================================================
@torch.no_grad()
def run_evaluation(
    unet: nn.Module,
    condition_encoder: nn.Module,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    logger: Logger,
    *,
    use_ddim: bool = False,
    max_samples: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    分块走完整反向采样, 累积所有逐样本指标。

    Args:
        loader:     val 或 test 的 DataLoader
        use_ddim:   True 时用 DDIM (dev 调试用), False 走 DDPM 完整链
        max_samples: 只评估前 N 个样本 (dev 调试用)

    Returns:
        dict, 含逐样本指标张量:
            "snr_input":   [N] 输入 SNR (dB, 含 ±0 扰动 = 数据集 split=val 时无扰动)
            "nmse":        [N]
            "out_snr_db":  [N]
            "in_snr_db":   [N]
            "snr_gain_db": [N]
    """
    unet.eval()
    condition_encoder.eval()

    method = "ddim" if use_ddim else "ddpm"
    if use_ddim:
        n_steps = cfg["eval"]["ddim_steps"]
        logger.info(f"[eval] 采样方法: DDIM (steps={n_steps}) — DEV 模式, 不进论文")
    else:
        n_steps = cfg["eval"]["full_sampling_steps"]
        logger.info(f"[eval] 采样方法: DDPM 完整链 (steps={n_steps})")

    snr_in_chunks: list[torch.Tensor] = []
    nmse_chunks:   list[torch.Tensor] = []
    out_db_chunks: list[torch.Tensor] = []
    in_db_chunks:  list[torch.Tensor] = []

    n_seen = 0
    t0 = time.time()
    for batch_i, batch in enumerate(loader):
        y, x_0, snr_db = batch
        y      = y.to(device, non_blocking=True)
        x_0    = x_0.to(device, non_blocking=True)
        snr_db = snr_db.to(device, non_blocking=True)
        B = y.size(0)

        # 走反向采样
        x_hat = sample(
            unet, condition_encoder, schedule,
            y=y, snr_db=snr_db,
            method=method,
            num_inference_steps=n_steps if use_ddim else None,  # DDPM 用全 T
            psd_refine_interval=cfg["eval"]["psd_refine_interval"],
            psd_nperseg=cfg["psd"]["nperseg"],
            psd_noverlap=cfg["psd"]["noverlap"],
            psd_eps=cfg["psd"]["eps"],
            progress=False,
        )  # [B, 2, L]

        m = compute_eval_metrics(x_hat, x_0, y=y)
        snr_in_chunks.append(snr_db.detach().cpu())
        nmse_chunks.append(m["nmse_per_sample"].detach().cpu())
        out_db_chunks.append(m["out_snr_db_per_sample"].detach().cpu())
        in_db_chunks.append(m["in_snr_db_per_sample"].detach().cpu())

        n_seen += B
        elapsed = time.time() - t0
        logger.info(
            f"[eval] batch {batch_i+1} (累计 {n_seen} 样本): "
            f"batch NMSE={m['nmse_mean'].item():.4f}  "
            f"OutSNR={m['out_snr_db_mean'].item():.2f}dB  "
            f"耗时 {elapsed:.1f}s"
        )

        if max_samples is not None and n_seen >= max_samples:
            logger.info(f"[eval] 达到 max_samples={max_samples}, 提前停止")
            break

    snr_in_all   = torch.cat(snr_in_chunks)
    nmse_all     = torch.cat(nmse_chunks)
    out_db_all   = torch.cat(out_db_chunks)
    in_db_all    = torch.cat(in_db_chunks)
    gain_db_all  = out_db_all - in_db_all

    # 必要时再裁剪到 max_samples (最后一批可能超)
    if max_samples is not None and snr_in_all.numel() > max_samples:
        snr_in_all  = snr_in_all[:max_samples]
        nmse_all    = nmse_all[:max_samples]
        out_db_all  = out_db_all[:max_samples]
        in_db_all   = in_db_all[:max_samples]
        gain_db_all = gain_db_all[:max_samples]

    return {
        "snr_input":   snr_in_all,    # [N]
        "nmse":        nmse_all,      # [N]
        "out_snr_db":  out_db_all,    # [N]
        "in_snr_db":   in_db_all,     # [N]
        "snr_gain_db": gain_db_all,   # [N]
    }


# ============================================================================
# 输出报告
# ============================================================================
def build_summary(
    results: dict[str, torch.Tensor],
    snr_bins: list[float] = [-15, -10, -5, 0, 5, 10],
) -> dict[str, Any]:
    """
    汇总总体均值 + 按 SNR 档位分组的均值, 返回可序列化 dict。
    """
    n_samples = int(results["nmse"].numel())
    overall = {
        "n_samples":         n_samples,
        "nmse_mean":         float(results["nmse"].mean().item()),
        "out_snr_db_mean":   float(results["out_snr_db"].mean().item()),
        "in_snr_db_mean":    float(results["in_snr_db"].mean().item()),
        "snr_gain_db_mean":  float(results["snr_gain_db"].mean().item()),
    }

    # 按 SNR 档位分组 (容差 ±2.5dB, 与训练 ±2dB 扰动兼容)
    nmse_by_snr   = group_by_snr(results["snr_input"], results["nmse"], snr_bins)
    out_db_by_snr = group_by_snr(results["snr_input"], results["out_snr_db"], snr_bins)
    gain_by_snr   = group_by_snr(results["snr_input"], results["snr_gain_db"], snr_bins)

    by_snr: dict[float, dict[str, float]] = {}
    for s in snr_bins:
        if s in nmse_by_snr:
            mask = (results["snr_input"] - s).abs() <= 2.5
            by_snr[float(s)] = {
                "n":            int(mask.sum().item()),
                "nmse":         float(nmse_by_snr[s].item()),
                "out_snr_db":   float(out_db_by_snr[s].item()),
                "snr_gain_db":  float(gain_by_snr[s].item()),
            }

    return {"overall": overall, "by_snr_db": by_snr}


def save_per_sample_csv(results: dict[str, torch.Tensor], csv_path: Path) -> None:
    """
    输出逐样本 CSV, 列: idx, snr_input, nmse, out_snr_db, in_snr_db, snr_gain_db
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "snr_input", "nmse", "out_snr_db", "in_snr_db", "snr_gain_db"])
        N = results["nmse"].numel()
        for i in range(N):
            w.writerow([
                i,
                f"{results['snr_input'][i].item():.4f}",
                f"{results['nmse'][i].item():.6f}",
                f"{results['out_snr_db'][i].item():.4f}",
                f"{results['in_snr_db'][i].item():.4f}",
                f"{results['snr_gain_db'][i].item():.4f}",
            ])


# ============================================================================
# 主入口
# ============================================================================
def main(
    config_path: str | Path = "configs/default.yaml",
    ckpt_path: Optional[str | Path] = None,
    *,
    split: str = "val",
    use_ddim: bool = False,
    max_samples: Optional[int] = None,
    save_per_sample: bool = False,
    eval_name: str = "eval",
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    主评估入口, 返回 summary dict (供 Notebook 检查)。

    Args:
        config_path:     yaml 路径
        ckpt_path:       ckpt 路径; None 时自动取 <run_dir>/ckpts/ckpt_best.pt
        split:           "val" | "all"  (val 用 dataset 后 val_ratio, all 全量)
        use_ddim:        True 时用 DDIM 加速 (dev 调试, 不进论文)
        max_samples:     只评估前 N 个 (dev 调试)
        save_per_sample: True 时输出 per-sample CSV
        eval_name:       输出子目录名 <run_dir>/eval/<eval_name>/
        overrides:       配置点式覆盖

    Returns:
        summary dict, 含 overall + by_snr_db 两段
    """
    # ---- 0. 配置 ----
    cfg = load_config(config_path, overrides=overrides)
    set_seed(cfg["misc"]["seed"])

    device_str = cfg["misc"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # ---- 1. run 目录 / 日志 ----
    paths = setup_run_dir(cfg["paths"]["output_root"], cfg["paths"]["exp_name"])
    eval_dir = paths["run_dir"] / "eval" / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(eval_dir / "eval_log.txt")
    logger.info(f"=== PC-CDDM 评估 ===  device={device}  eval_dir={eval_dir}")

    # ---- 2. ckpt ----
    if ckpt_path is None:
        ckpt_mgr = CheckpointManager(paths["ckpt_dir"])
        ckpt_path = ckpt_mgr.find_best()
        if ckpt_path is None:
            raise FileNotFoundError(
                f"未指定 ckpt 且 {paths['ckpt_dir']}/ckpt_best.pt 不存在"
            )
        logger.info(f"[ckpt] 自动选择 best: {ckpt_path}")
    else:
        ckpt_path = Path(ckpt_path)
        logger.info(f"[ckpt] 使用指定路径: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(
        f"[ckpt] epoch={state.get('epoch', '?')}  "
        f"best_val_nmse={state.get('best_val_nmse', '?')}"
    )

    # ---- 3. 数据 ----
    eval_split = "all" if split == "all" else "val"
    eval_ds = IQDataset.from_config(cfg["data"], split=eval_split)
    logger.info(f"[data] split={eval_split}  N={len(eval_ds)}")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ---- 4. 模型 / 调度 ----
    unet = UNet1D.from_config(cfg["model"]).to(device)
    condition_encoder = ConditionEncoder.from_config(cfg["model"], cfg["psd"]).to(device)
    schedule = DiffusionSchedule.from_config(cfg["diffusion"]).to(device)

    unet.load_state_dict(state["unet_state"])
    condition_encoder.load_state_dict(state["ce_state"])
    logger.info("[model] 权重加载完成")

    # ---- 5. 跑评估 ----
    results = run_evaluation(
        unet, condition_encoder, schedule, eval_loader, cfg, device, logger,
        use_ddim=use_ddim, max_samples=max_samples,
    )

    # ---- 6. 汇总 + 输出 ----
    summary = build_summary(results)

    # 在 summary 内附带元信息
    summary["meta"] = {
        "ckpt_path":   str(ckpt_path),
        "config_path": str(config_path),
        "split":       split,
        "use_ddim":    use_ddim,
        "max_samples": max_samples,
        "device":      str(device),
        "ckpt_epoch":  state.get("epoch"),
        "ckpt_best_val_nmse": state.get("best_val_nmse"),
    }

    # 写 summary yaml
    summary_path = eval_dir / "eval_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    logger.info(f"[output] summary 写入 {summary_path}")

    # 写 per-sample CSV (可选)
    if save_per_sample:
        csv_path = eval_dir / "eval_per_sample.csv"
        save_per_sample_csv(results, csv_path)
        logger.info(f"[output] per-sample CSV 写入 {csv_path}")

    # ---- 7. 控制台总览 ----
    logger.info("=" * 60)
    logger.info("评估总览:")
    overall = summary["overall"]
    logger.info(f"  N={overall['n_samples']}")
    logger.info(f"  NMSE      (overall): {overall['nmse_mean']:.4f}")
    logger.info(f"  OutSNR dB (overall): {overall['out_snr_db_mean']:.2f}")
    logger.info(f"  InSNR  dB (overall): {overall['in_snr_db_mean']:.2f}")
    logger.info(f"  Gain   dB (overall): {overall['snr_gain_db_mean']:.2f}")
    logger.info("按输入 SNR 档位:")
    for snr, m in summary["by_snr_db"].items():
        logger.info(
            f"  SNR={snr:>5.1f} dB  N={m['n']:>4d}  "
            f"NMSE={m['nmse']:.4f}  OutSNR={m['out_snr_db']:.2f}dB  "
            f"Gain={m['snr_gain_db']:.2f}dB"
        )
    logger.info("=" * 60)
    logger.close()

    return summary


# ============================================================================
# 命令行
# ============================================================================
if __name__ == "__main__":
    from pc_cddm.train import _parse_overrides

    parser = argparse.ArgumentParser(description="PC-CDDM 评估")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt",   default=None,
                        help="ckpt 路径; 不指定时取 <run_dir>/ckpts/ckpt_best.pt")
    parser.add_argument("--split",  default="val", choices=["val", "all"])
    parser.add_argument("--use_ddim", action="store_true",
                        help="使用 DDIM 加速 (DEV 模式, 不进论文)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="只评估前 N 个样本 (DEV 模式)")
    parser.add_argument("--save_per_sample", action="store_true",
                        help="输出逐样本 CSV")
    parser.add_argument("--eval_name", default="eval",
                        help="输出子目录名 (区分多次评估), e.g. eval_ddpm_1000")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    overrides = _parse_overrides(args.override) if args.override else None
    main(
        args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        use_ddim=args.use_ddim,
        max_samples=args.max_samples,
        save_per_sample=args.save_per_sample,
        eval_name=args.eval_name,
        overrides=overrides,
    )
