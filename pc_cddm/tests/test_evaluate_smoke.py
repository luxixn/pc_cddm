"""
evaluate.py 端到端自检 (微型规模)。

策略:
    1. 复用 test_train_smoke 的微型配置, 先训 1 epoch 出一个微型 ckpt
    2. 跑 evaluate(use_ddim=True), 验证流程通畅 + 文件输出正确
    3. 跑 evaluate(use_ddim=False, max_samples=4) 验证 DDPM 全链路也能工作
    4. 验证 summary yaml 结构、per-sample CSV 行数

不验证 NMSE 数值, 仅验证流程不报错、状态正确。

运行: python -m pc_cddm.tests.test_evaluate_smoke
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import yaml
from pathlib import Path

import torch

from pc_cddm.train import main as train_main
from pc_cddm.evaluate import main as eval_main
from pc_cddm.tests.test_train_smoke import make_mock_h5, make_micro_cfg, _run_with_cfg


def _train_a_micro_ckpt(work: Path) -> tuple[str, Path]:
    """
    用微型配置训 1 epoch, 返回 (yaml 路径, ckpt 目录的 best.pt 路径)。
    """
    h5_path = work / "mock.h5"
    make_mock_h5(str(h5_path), N=30, L=1024)
    output_root = work / "runs"

    cfg = make_micro_cfg(str(h5_path), str(output_root),
                         exp_name="eval_smoke", num_epochs=1)
    cfg_path = work / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    summary = train_main(str(cfg_path))
    ckpt_dir = Path(summary["run_dir"]) / "ckpts"
    best_ckpt = ckpt_dir / "ckpt_best.pt"
    assert best_ckpt.exists(), f"训练后未生成 best ckpt: {best_ckpt}"
    return str(cfg_path), best_ckpt


# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    work = Path(tempfile.mkdtemp(prefix="pccddm_eval_smoke_"))
    print(f"工作目录: {work}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("准备阶段: 训 1 个微型 ckpt")
    print("=" * 60)
    # ==================================================================
    cfg_path, ckpt_path = _train_a_micro_ckpt(work)
    print(f"  cfg : {cfg_path}")
    print(f"  ckpt: {ckpt_path}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 1: evaluate(use_ddim=True) 在 val 上跑通")
    print("=" * 60)
    # ==================================================================
    summary = eval_main(
        cfg_path,
        ckpt_path=str(ckpt_path),
        split="val",
        use_ddim=True,
        save_per_sample=True,
        eval_name="ddim_smoke",
    )
    print(f"\nsummary 结构:")
    print(f"  顶层键: {list(summary.keys())}")
    print(f"  overall: {list(summary['overall'].keys())}")
    print(f"  by_snr_db 档位: {list(summary['by_snr_db'].keys())}")
    print(f"  meta.use_ddim: {summary['meta']['use_ddim']}")

    # 文件检查
    eval_dir = Path(summary["meta"]["ckpt_path"]).parent.parent / "eval" / "ddim_smoke"
    files = sorted(p.name for p in eval_dir.iterdir())
    print(f"  eval_dir 文件: {files}")

    has_summary = (eval_dir / "eval_summary.yaml").exists()
    has_csv     = (eval_dir / "eval_per_sample.csv").exists()
    has_log     = (eval_dir / "eval_log.txt").exists()
    print(f"  summary yaml ✓: {has_summary}")
    print(f"  per-sample CSV ✓: {has_csv}")
    print(f"  log ✓: {has_log}")

    # CSV 行数
    if has_csv:
        with (eval_dir / "eval_per_sample.csv").open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        n_rows = len(rows) - 1  # 减去 header
        print(f"  CSV 行数 (除 header): {n_rows}  (期望 6, val 6 个样本)")
        print(f"  CSV header: {rows[0]}")
        print(f"  ✓: {n_rows == 6}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: evaluate(use_ddim=False, max_samples=4) DDPM 全链 + 提前停")
    print("=" * 60)
    # ==================================================================
    summary2 = eval_main(
        cfg_path,
        ckpt_path=str(ckpt_path),
        split="all",
        use_ddim=False,
        max_samples=4,
        eval_name="ddpm_smoke",
    )
    print(f"  meta.use_ddim: {summary2['meta']['use_ddim']}")
    print(f"  meta.max_samples: {summary2['meta']['max_samples']}")
    print(f"  overall.n_samples: {summary2['overall']['n_samples']}  (期望 4)")
    print(f"  ✓: {summary2['overall']['n_samples'] == 4}")
    print(f"  use_ddim=False ✓: {summary2['meta']['use_ddim'] is False}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: ckpt_path=None 自动找 best")
    print("=" * 60)
    # ==================================================================
    summary3 = eval_main(
        cfg_path,
        ckpt_path=None,                # 自动找 ckpt_best.pt
        split="val",
        use_ddim=True,
        max_samples=2,
        eval_name="auto_best_smoke",
    )
    auto_path = summary3["meta"]["ckpt_path"]
    print(f"  自动选中 ckpt: {auto_path}")
    print(f"  指向 best 文件 ✓: {Path(auto_path).name == 'ckpt_best.pt'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: summary yaml 内容可读且字段正确")
    print("=" * 60)
    # ==================================================================
    summary_path = eval_dir / "eval_summary.yaml"
    with summary_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    print(f"  yaml 顶层键: {list(loaded.keys())}")
    expected_keys = {"overall", "by_snr_db", "meta"}
    print(f"  字段完整 ✓: {set(loaded.keys()) == expected_keys}")

    overall_keys_expected = {"n_samples", "nmse_mean", "out_snr_db_mean",
                             "in_snr_db_mean", "snr_gain_db_mean"}
    print(f"  overall 字段完整 ✓: "
          f"{set(loaded['overall'].keys()) == overall_keys_expected}")

    # 每个 SNR 档位都应有 4 个字段
    by_snr_ok = all(
        set(v.keys()) == {"n", "nmse", "out_snr_db", "snr_gain_db"}
        for v in loaded["by_snr_db"].values()
    )
    print(f"  by_snr_db 各档字段完整 ✓: {by_snr_ok}")

    # ---- 清理 ----
    print("\n" + "=" * 60)
    print(f"清理工作目录: {work}")
    print("=" * 60)
    shutil.rmtree(work, ignore_errors=True)
    print("\n全部自检完成")
