"""
PC-CDDM 评估指标。

核心指标:
    NMSE (Normalized MSE):
        逐样本 NMSE_i = ||x̂_i - x_i||² / ||x_i||²
        全批次 NMSE = mean_i NMSE_i
        值域 [0, +∞), 越小越好, 完美预测 NMSE = 0

    Output SNR (dB):
        逐样本 OutSNR_i = 10·log10(||x_i||² / ||x̂_i - x_i||²) = -10·log10(NMSE_i)
        值越大越好, 完美预测 OutSNR = +∞

设计要点:
    1. 全部支持 [B, C, L] 输入, 在 (C, L) 维上求范数, 留 batch 维。
    2. 同时返回逐样本张量与标量均值, 便于按 SNR 档位分组统计。
    3. 数值稳定: 分母 ||x_gt||² 加 eps; OutSNR 在 NMSE 极小时 clamp 上限避免 inf。
"""

from __future__ import annotations

import torch


# ============================================================================
# 基础指标
# ============================================================================
def nmse(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    *,
    eps: float = 1e-12,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Normalized MSE: ||x̂ - x||² / ||x||²。

    Args:
        x_hat: [B, C, L] 预测信号
        x_gt:  [B, C, L] 干净信号 (ground truth)
        eps:   分母数值保护
        reduce: True 返回标量 (batch 均值), False 返回 [B] 逐样本

    Returns:
        scalar 或 [B]
    """
    if x_hat.shape != x_gt.shape:
        raise ValueError(
            f"shape 不匹配: x_hat={tuple(x_hat.shape)}, x_gt={tuple(x_gt.shape)}"
        )
    if x_hat.dim() != 3:
        raise ValueError(f"期望 [B, C, L], 实际 {tuple(x_hat.shape)}")

    # 在 (C, L) 维度上求平方和, 留 batch 维 [B]
    err_sq = (x_hat - x_gt).pow(2).flatten(1).sum(dim=-1)  # [B]
    sig_sq = x_gt.pow(2).flatten(1).sum(dim=-1)            # [B]
    per_sample = err_sq / (sig_sq + eps)                   # [B]

    return per_sample.mean() if reduce else per_sample


def output_snr_db(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    *,
    eps: float = 1e-12,
    max_db: float = 100.0,
    reduce: bool = True,
) -> torch.Tensor:
    """
    输出 SNR (dB): 10·log10(||x||² / ||x̂ - x||²) = -10·log10(NMSE)。

    Args:
        x_hat:  [B, C, L]
        x_gt:   [B, C, L]
        eps:    数值保护
        max_db: 完美预测时 NMSE→0 会让 dB→inf, 上限 clamp 至 max_db
        reduce: True 返回标量, False 返回 [B]

    Returns:
        scalar 或 [B]
    """
    per_sample_nmse = nmse(x_hat, x_gt, eps=eps, reduce=False)  # [B]
    # log10(0) 数值保护: nmse 加 eps; 同时 clamp 上限
    per_sample_db = -10.0 * torch.log10(per_sample_nmse + eps)
    per_sample_db = per_sample_db.clamp(max=max_db)

    return per_sample_db.mean() if reduce else per_sample_db


def input_snr_db(
    y: torch.Tensor,
    x_gt: torch.Tensor,
    *,
    eps: float = 1e-12,
    max_db: float = 100.0,
    reduce: bool = True,
) -> torch.Tensor:
    """
    输入 SNR (dB): 10·log10(||x||² / ||y - x||²)。

    可用于核对样本实际 SNR 与文件名 SNR 是否一致 (调试/可视化用)。

    Args:
        y:    [B, C, L] 含噪观测
        x_gt: [B, C, L] 干净信号
        eps / max_db / reduce: 同上
    """
    return output_snr_db(y, x_gt, eps=eps, max_db=max_db, reduce=reduce)


# ============================================================================
# 一站式指标聚合
# ============================================================================
def compute_eval_metrics(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    *,
    y: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    一站式返回所有评估指标 (标量均值 + 逐样本张量)。

    Args:
        x_hat: [B, C, L] 模型输出
        x_gt:  [B, C, L] 干净信号
        y:     [B, C, L] 含噪观测 (可选, 提供则计算 input_snr 与 SNR 提升量)

    Returns:
        dict, 含:
            "nmse_mean":         scalar
            "nmse_per_sample":   [B]
            "out_snr_db_mean":   scalar
            "out_snr_db_per_sample": [B]
            (若 y 提供:)
            "in_snr_db_mean":    scalar
            "in_snr_db_per_sample":  [B]
            "snr_gain_db_mean":  scalar  (= out - in, 单位 dB)
            "snr_gain_db_per_sample": [B]
    """
    nmse_per = nmse(x_hat, x_gt, reduce=False)              # [B]
    out_db_per = output_snr_db(x_hat, x_gt, reduce=False)   # [B]

    out: dict[str, torch.Tensor] = {
        "nmse_mean":             nmse_per.mean(),
        "nmse_per_sample":       nmse_per,
        "out_snr_db_mean":       out_db_per.mean(),
        "out_snr_db_per_sample": out_db_per,
    }

    if y is not None:
        in_db_per = input_snr_db(y, x_gt, reduce=False)  # [B]
        gain_per = out_db_per - in_db_per                # [B]
        out.update({
            "in_snr_db_mean":         in_db_per.mean(),
            "in_snr_db_per_sample":   in_db_per,
            "snr_gain_db_mean":       gain_per.mean(),
            "snr_gain_db_per_sample": gain_per,
        })

    return out


# ============================================================================
# 按 SNR 档位分组聚合 (论文表格用)
# ============================================================================
def group_by_snr(
    snr_db: torch.Tensor,            # [B] 物理 SNR 标量 (dB), 输入端
    metric_per_sample: torch.Tensor, # [B] 任意逐样本指标 (NMSE / OutSNR)
    snr_bins: list[float] | None = None,
) -> dict[float, torch.Tensor]:
    """
    将逐样本指标按 SNR 档位分组取均值。

    Args:
        snr_db:            [B] 输入 SNR (dB)
        metric_per_sample: [B] 任意标量指标
        snr_bins:          目标 SNR 档位 (例 [-15, -10, -5, 0, 5, 10]),
                           None 时取 snr_db 中所有出现过的整数值

    Returns:
        dict[snr_value -> 该档位的均值], snr_value 为 float
    """
    if snr_db.shape != metric_per_sample.shape:
        raise ValueError(
            f"shape 不匹配: snr_db={tuple(snr_db.shape)}, "
            f"metric={tuple(metric_per_sample.shape)}"
        )

    if snr_bins is None:
        # 取 snr_db 出现过的整数 SNR (容差 0.5dB, 训练扰动 ±2dB 仍可归位)
        unique_snrs = torch.unique(snr_db.round()).tolist()
        snr_bins = sorted(unique_snrs)

    result: dict[float, torch.Tensor] = {}
    for snr in snr_bins:
        # 容差 ±2.5dB (留出 ±2dB 训练扰动余量)
        mask = (snr_db - snr).abs() <= 2.5
        if mask.any():
            result[float(snr)] = metric_per_sample[mask].mean()

    return result


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.utils.metrics
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # ==================================================================
    print("=" * 60)
    print("测试 1: 完美预测 NMSE = 0, OutSNR = max_db")
    print("=" * 60)
    # ==================================================================
    B, C, L = 4, 2, 1024
    x_gt = torch.randn(B, C, L)
    x_hat_perfect = x_gt.clone()

    nm = nmse(x_hat_perfect, x_gt)
    db = output_snr_db(x_hat_perfect, x_gt, max_db=100.0)
    print(f"NMSE        = {nm.item():.4e}  (期望 ~ 0)")
    print(f"OutSNR (dB) = {db.item():.2f}  (期望 100.0, 即 max_db clamp)")
    print(f"NMSE  ✓: {nm.item() < 1e-10}")
    print(f"OutSNR ✓: {abs(db.item() - 100.0) < 1e-3}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: 已知 NMSE 数值验证")
    print("=" * 60)
    # ==================================================================
    # 构造: x_hat = 0, 则 ||x_hat - x||² = ||x||², NMSE = 1.0, OutSNR = 0 dB
    x_hat_zero = torch.zeros_like(x_gt)
    nm = nmse(x_hat_zero, x_gt)
    db = output_snr_db(x_hat_zero, x_gt)
    print(f"x̂=0:  NMSE = {nm.item():.6f}  (期望 1.0)")
    print(f"      OutSNR = {db.item():.4f} dB  (期望 0.0)")
    print(f"  ✓: {abs(nm.item() - 1.0) < 1e-5 and abs(db.item()) < 1e-3}")

    # 构造: x_hat = 0.9 * x_gt, 则 ||x_hat - x||² = 0.01 * ||x||², NMSE = 0.01
    # OutSNR = -10 * log10(0.01) = 20 dB
    x_hat_9 = 0.9 * x_gt
    nm = nmse(x_hat_9, x_gt)
    db = output_snr_db(x_hat_9, x_gt)
    print(f"x̂=0.9x: NMSE = {nm.item():.6f}  (期望 0.01)")
    print(f"        OutSNR = {db.item():.4f} dB  (期望 20.0)")
    print(f"  ✓: {abs(nm.item() - 0.01) < 1e-4 and abs(db.item() - 20.0) < 1e-2}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: reduce=False 返回逐样本 [B]")
    print("=" * 60)
    # ==================================================================
    x_hat_noisy = x_gt + 0.3 * torch.randn_like(x_gt)
    nm_per = nmse(x_hat_noisy, x_gt, reduce=False)
    db_per = output_snr_db(x_hat_noisy, x_gt, reduce=False)
    print(f"NMSE per-sample shape:   {tuple(nm_per.shape)}  (期望 [{B}])")
    print(f"OutSNR per-sample shape: {tuple(db_per.shape)}  (期望 [{B}])")
    print(f"NMSE per-sample: {nm_per.tolist()}")
    print(f"OutSNR per-sample (dB): {[f'{x:.2f}' for x in db_per.tolist()]}")
    print(f"  shape ✓: {nm_per.shape == (B,) and db_per.shape == (B,)}")
    print(f"  mean 一致 ✓: {abs(nm_per.mean().item() - nmse(x_hat_noisy, x_gt).item()) < 1e-6}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: 输入 SNR 校核 (用真实噪声构造已知输入 SNR)")
    print("=" * 60)
    # ==================================================================
    # 构造 -10 dB 输入: ||n||² = 10 * ||x||², NMSE_in = 10, in_snr = -10 dB
    x_clean = torch.randn(B, C, L)
    sig_pow = x_clean.pow(2).mean()
    target_in_snr_db = -10.0
    noise_pow = sig_pow * 10 ** (-target_in_snr_db / 10)  # 噪声功率
    n = torch.randn_like(x_clean) * noise_pow.sqrt()
    y = x_clean + n
    in_snr = input_snr_db(y, x_clean)
    print(f"目标输入 SNR: {target_in_snr_db:.1f} dB")
    print(f"实测输入 SNR: {in_snr.item():.2f} dB")
    print(f"  ✓ (容差 ±1dB): {abs(in_snr.item() - target_in_snr_db) < 1.0}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 5: compute_eval_metrics 一站式")
    print("=" * 60)
    # ==================================================================
    metrics = compute_eval_metrics(x_hat_noisy, x_gt, y=y)
    print("返回字段:")
    for k, v in metrics.items():
        if v.dim() == 0:
            print(f"  {k:30s} = {v.item():.4f}  (scalar)")
        else:
            print(f"  {k:30s} shape={tuple(v.shape)}  ({v.dtype})")

    expected_keys = {"nmse_mean", "nmse_per_sample", "out_snr_db_mean",
                     "out_snr_db_per_sample", "in_snr_db_mean",
                     "in_snr_db_per_sample", "snr_gain_db_mean",
                     "snr_gain_db_per_sample"}
    print(f"  字段完整 ✓: {set(metrics.keys()) == expected_keys}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 6: 按 SNR 档位分组")
    print("=" * 60)
    # ==================================================================
    # 模拟批次 SNR (含 ±2 扰动): -15 -15 -10 -10 -5 -5 0 0 5 5 10 10
    snr_db = torch.tensor([-15.3, -14.8, -10.2, -9.7, -5.1, -4.9,
                           -0.2,   0.3,   4.7,   5.4, 10.2,  9.6])
    metric_per = torch.tensor([0.20, 0.22, 0.10, 0.11, 0.05, 0.06,
                                0.03, 0.03, 0.01, 0.01, 0.005, 0.006])
    grouped = group_by_snr(snr_db, metric_per, snr_bins=[-15, -10, -5, 0, 5, 10])
    print("按整数 SNR 分组的 NMSE 均值:")
    for snr, val in grouped.items():
        print(f"  SNR={snr:>5.1f} dB: NMSE = {val.item():.4f}")
    print(f"  ✓ 6 个档位都有: {len(grouped) == 6}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 7: shape mismatch 报错")
    print("=" * 60)
    # ==================================================================
    try:
        nmse(torch.randn(2, 2, 1024), torch.randn(2, 2, 512))
        print("✗ 应抛 ValueError 未抛出")
    except ValueError as e:
        print(f"✓ shape 不匹配正确报错: {e}")

    print("\n" + "=" * 60)
    print("全部自检完成")
    print("=" * 60)
