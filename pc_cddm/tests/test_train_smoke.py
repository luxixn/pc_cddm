"""
train.py 端到端自检 (微型规模)。

策略:
    用 30 个样本的 mock H5 + T=20 的微型 schedule + base_channels=8 的微型 UNet
    跑 2 个 epoch, 验证训练循环、ckpt、续训、wallclock 退出全部通畅。
    不验证 NMSE 数值, 仅验证流程不报错、状态正确。

运行: python -m pc_cddm.tests.test_train_smoke
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from pc_cddm.train import main, load_config, _parse_overrides


# ============================================================================
# 工具: 制作 mock H5 数据集
# ============================================================================
def make_mock_h5(path: str, N: int = 30, L: int = 1024) -> None:
    """生成 N 个样本的 mock H5, SNR 在 -15..10 间循环。"""
    snr_vals = [-15, -10, -5, 0, 5, 10]

    def to_str(v: int) -> str:
        return f"n{abs(v)}" if v < 0 else f"p{v}"

    fns = []
    for i in range(N):
        s = snr_vals[i % len(snr_vals)]
        fns.append(f"HZ_DL_LFM_{to_str(s)}_GS_{i+1:05d}.mat".encode())

    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("filenames", data=np.array(fns, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
        f.create_dataset("hz_signals", data=rng.standard_normal((N, 2, L)).astype(np.float32))
        f.create_dataset("wz_signals", data=rng.standard_normal((N, 2, L)).astype(np.float32))


# ============================================================================
# 工具: 微型配置 (覆盖默认 yaml 关键字段)
# ============================================================================
def make_micro_cfg(h5_path: str, output_root: str, exp_name: str = "smoke",
                   max_hours: float = 1.0, num_epochs: int = 2) -> dict:
    """组装一份微型规模的配置, 直接传给 main()。"""
    return {
        "paths": {
            "data": h5_path,
            "output_root": output_root,
            "exp_name": exp_name,
            "resume_from": None,
        },
        "data": {
            "h5_path": h5_path,  # main() 内部还会重置一次, 此处冗余无害
            "signal_length": 1024, "num_channels": 2,
            "val_ratio": 0.2,
            "snr_perturb_db": 2.0,
            "preload": True, "seed": 42,
            "num_workers": 0, "pin_memory": False,
        },
        "diffusion": {
            "num_timesteps": 20,            # 微型 T
            "beta_schedule": "linear",
            "beta_start": 1e-4, "beta_end": 0.02,
            "psd_loss_threshold_ratio": 0.25,
        },
        "psd": {"nperseg": 256, "noverlap": 128, "fs": 1.0,
                "eps": 1e-8, "detrend": True},
        "model": {
            "time_embed_dim": 32, "time_mlp_dim": 16,
            "snr_embed_dim": 32, "snr_mlp_dim": 16,
            "snr_encoding": "sinusoidal",
            "snr_min": -15.0, "snr_max": 10.0,
            "psd_mlp_hidden": 64, "psd_mlp_out": 16,
            "cond_dim": 48,
            "base_channels": 8, "channel_mults": [1, 2],
            "num_res_blocks": 1, "groupnorm_groups": 4,
        },
        "train": {
            "batch_size": 4, "num_epochs": num_epochs,
            "lr": 2e-4, "weight_decay": 0.0,
            "grad_clip": 1.0, "grad_accum_steps": 1,
            "amp": False,                   # CPU 自检关掉 AMP
            "lambda_psd": 0.1,
            "log_every": 5,
            "val_every_epochs": 1,
            "save_every_epochs": 1,
            "keep_last_n_ckpts": 3,
            "max_wallclock_hours": max_hours,
            "val_ddim_steps": 5,           # 微型 DDIM 步数
        },
        "eval": {
            "full_sampling_steps": 20,
            "psd_refine_interval": 5,
            "use_ddim": True, "ddim_steps": 5,
            "batch_size": 4,
            "chunk_size": 100,
        },
        "misc": {"seed": 42, "device": "cpu", "pythonhashseed": 42},
    }


# ============================================================================
# 测试: 把内存 cfg 持久化到临时 yaml 后再调用 main
# ============================================================================
def _run_with_cfg(cfg: dict, work_dir: Path) -> dict:
    """把 cfg dump 成 yaml, 调 main(), 返回训练结束摘要。"""
    cfg_path = work_dir / "tmp_cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return main(str(cfg_path))


# ============================================================================
# 主自检
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    work = Path(tempfile.mkdtemp(prefix="pccddm_smoke_"))
    print(f"工作目录: {work}")

    h5_path = work / "mock.h5"
    make_mock_h5(str(h5_path), N=30, L=1024)
    output_root = work / "runs"

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 1: 从头训练 2 epoch")
    print("=" * 60)
    # ==================================================================
    cfg1 = make_micro_cfg(str(h5_path), str(output_root), exp_name="smoke1",
                          max_hours=1.0, num_epochs=2)
    summary = _run_with_cfg(cfg1, work)
    print(f"\n训练结束摘要:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    run_dir = Path(summary["run_dir"])
    ckpt_dir = run_dir / "ckpts"
    ckpts = sorted(ckpt_dir.iterdir())
    print(f"\nckpt 目录文件:")
    for c in ckpts:
        print(f"  {c.name}")

    print(f"\nepoch 全跑完: {'✓' if summary['epochs_completed'] == 2 else '✗'}")
    print(f"early_stop=None: {'✓' if summary['early_stop_reason'] is None else '✗'}")
    print(f"latest 存在: {'✓' if (ckpt_dir / 'ckpt_latest.pt').exists() else '✗'}")
    print(f"best   存在: {'✓' if (ckpt_dir / 'ckpt_best.pt').exists()   else '✗'}")
    print(f"日志文件存在: {'✓' if (run_dir / 'log.txt').exists() else '✗'}")
    print(f"配置存档存在: {'✓' if (run_dir / 'config.yaml').exists() else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: 断点续训 (resume_from=auto, 再跑到 epoch 4)")
    print("=" * 60)
    # ==================================================================
    cfg2 = make_micro_cfg(str(h5_path), str(output_root), exp_name="smoke1",
                          max_hours=1.0, num_epochs=4)
    cfg2["paths"]["resume_from"] = "auto"
    summary2 = _run_with_cfg(cfg2, work)
    print(f"\n续训摘要:")
    for k, v in summary2.items():
        print(f"  {k}: {v}")

    print(f"\n续训跑到 epoch 4: {'✓' if summary2['epochs_completed'] == 4 else '✗'}")
    print(f"global_step 单调增 (>{summary['global_step']}): "
          f"{'✓' if summary2['global_step'] > summary['global_step'] else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: wallclock 退出 (max_hours=1秒, 训练应在第 1 epoch 后退出)")
    print("=" * 60)
    # ==================================================================
    cfg3 = make_micro_cfg(str(h5_path), str(output_root), exp_name="smoke3",
                          max_hours=1.0 / 3600.0, num_epochs=10)  # 1 秒上限
    summary3 = _run_with_cfg(cfg3, work)
    print(f"\nwallclock 测试摘要:")
    for k, v in summary3.items():
        print(f"  {k}: {v}")

    early_stop = summary3["early_stop_reason"] == "wallclock_exceeded"
    not_all = summary3["epochs_completed"] < 10
    print(f"\nearly_stop=wallclock: {'✓' if early_stop else '✗'}")
    print(f"未跑完所有 epoch: {'✓' if not_all else '✗'}")

    # 退出后状态可继续从 latest 续训
    cfg3_resume = dict(cfg3)
    cfg3_resume["paths"] = dict(cfg3["paths"])
    cfg3_resume["paths"]["resume_from"] = "auto"
    cfg3_resume["train"] = dict(cfg3["train"])
    cfg3_resume["train"]["max_wallclock_hours"] = 1.0  # 恢复正常
    cfg3_resume["train"]["num_epochs"] = summary3["epochs_completed"] + 1
    summary3b = _run_with_cfg(cfg3_resume, work)
    print(f"\nwallclock 退出后续训摘要:")
    print(f"  epochs_completed: {summary3b['epochs_completed']} "
          f"(期望 {summary3['epochs_completed'] + 1})")
    print(f"  续训成功: "
          f"{'✓' if summary3b['epochs_completed'] == summary3['epochs_completed'] + 1 else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: load_config 解析 yaml + h5_path 注入")
    print("=" * 60)
    # ==================================================================
    cfg_for_load = make_micro_cfg(str(h5_path), str(output_root))
    yaml_path = work / "load_test.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_for_load, f, allow_unicode=True)

    loaded = load_config(yaml_path)
    print(f"  data.h5_path 自动注入: '{loaded['data']['h5_path']}'")
    print(f"  与 paths.data 一致: "
          f"{'✓' if loaded['data']['h5_path'] == loaded['paths']['data'] else '✗'}")

    # 点式 override
    over = _parse_overrides(["train.lr=1e-3", "paths.exp_name=overridden"])
    loaded2 = load_config(yaml_path, overrides=over)
    print(f"  override train.lr: {loaded2['train']['lr']}  (期望 0.001)")
    print(f"  override exp_name: '{loaded2['paths']['exp_name']}'  (期望 'overridden')")
    print(f"  override 生效: "
          f"{'✓' if loaded2['train']['lr'] == 1e-3 and loaded2['paths']['exp_name'] == 'overridden' else '✗'}")

    # ---- 清理 ----
    print("\n" + "=" * 60)
    print(f"清理工作目录: {work}")
    print("=" * 60)
    shutil.rmtree(work, ignore_errors=True)
    print("\n全部自检完成")
