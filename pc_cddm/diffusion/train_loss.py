"""
PC-CDDM training loss: L_diff + 时间自适应 L_psd。

数学定义:
    L_total = L_diff + λ · 𝟙(t < T_threshold) · L_psd

    L_diff = MSE(ε̂, ε)                                      标准 DDPM 噪声预测损失
    L_psd  = MSE(log(psd(n̂) + ε), log(psd_gt + ε))           log 域 PSD MSE
    n̂ = y - x̂_0,  x̂_0 = predict_x0_from_eps(x_t, t, ε̂)

设计要点:
    1. 时间自适应: PSD 损失仅对 t < T_threshold (默认 T/4 = 250) 的样本施加。
       原因: t 大时 x̂_0 反推数值放大严重 (~ √(1/ᾱ_t)), PSD 物理约束不可靠。
    2. 逐样本 mask: 不是对整批用一个标量 mask, 而是按样本 t 值独立判定,
       然后只在 mask=True 的样本上累加 L_psd 贡献后取均值。
    3. 全批次都算 x̂_0 (不预先 mask): 为了让梯度反向传到 ε̂ 的所有样本。
       PSD 损失的贡献本身被 mask 屏蔽, 但 L_diff 对所有样本生效。
    4. PSD 真值由训练循环外部传入 (n_gt = y - x_0 -> Welch),
       因为同一份 psd_gt 既作为条件输入又作为损失目标, 共用计算。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_cddm.utils.psd import welch_psd_log
from pc_cddm.diffusion.schedule import DiffusionSchedule


@dataclass
class LossOutput:
    """训练循环里 logger 需要分别记录 L_diff / L_psd 趋势。"""
    total: torch.Tensor       # 标量, 用于 backward
    diff: torch.Tensor        # 标量, L_diff (detached for log)
    psd: torch.Tensor         # 标量, L_psd  (detached for log; 若该批无低 t 样本则为 0)
    n_psd_samples: int        # 该批次内施加 L_psd 的样本数 (用于 logger 显示)


def compute_pcddm_loss(
    eps_pred: torch.Tensor,       # [B, 2, L] 网络预测噪声 ε̂
    eps_target: torch.Tensor,     # [B, 2, L] 真实加入的噪声 ε
    x_t: torch.Tensor,            # [B, 2, L] 加噪信号
    y: torch.Tensor,              # [B, 2, L] 观测信号 y = x_0 + n
    t: torch.Tensor,              # [B] long 时间步
    schedule: DiffusionSchedule,
    psd_gt_log: torch.Tensor,     # [B, 256] 真实噪声 log PSD (训练循环算好传入)
    *,
    lambda_psd: float = 0.1,
    psd_loss_threshold: int = 250,    # T_threshold (默认 T/4)
    psd_nperseg: int = 256,
    psd_noverlap: int = 128,
    psd_fs: float = 1.0,
    psd_eps: float = 1e-8,
    psd_detrend: bool = True,
) -> LossOutput:
    """
    计算 PC-CDDM 总损失。

    Args:
        eps_pred:     [B, 2, L]  网络输出 ε̂
        eps_target:   [B, 2, L]  真实噪声 ε (训练时从 q_sample 采样)
        x_t:          [B, 2, L]  加噪信号
        y:            [B, 2, L]  观测信号 (= x_0 + n, 训练时已知)
        t:            [B]        时间步
        schedule:     DiffusionSchedule
        psd_gt_log:   [B, 256]   真实噪声 n=y-x_0 的 log PSD
        lambda_psd:   PSD 损失权重
        psd_loss_threshold: 仅 t < threshold 时施加 L_psd
        psd_*:        Welch 参数 (与 utils.psd 一致)

    Returns:
        LossOutput(total, diff, psd, n_psd_samples)
    """
    # 输入维度校验 (开发阶段保留, 生产可去掉)
    if eps_pred.shape != eps_target.shape:
        raise ValueError(
            f"eps_pred {tuple(eps_pred.shape)} 和 eps_target {tuple(eps_target.shape)} 形状不匹配"
        )
    if x_t.shape != y.shape:
        raise ValueError(f"x_t 和 y 形状不匹配")
    if t.dim() != 1 or t.size(0) != eps_pred.size(0):
        raise ValueError(f"t shape 期望 [B], 实际 {tuple(t.shape)}")

    B = eps_pred.size(0)

    # ===== L_diff: 标准 MSE =====
    # 在 [B, 2, L] 上逐元素 MSE 后取均值, 标量
    loss_diff = F.mse_loss(eps_pred, eps_target)

    # ===== L_psd: 时间自适应 =====
    # 1. 计算逐样本 mask
    mask = (t < psd_loss_threshold)  # [B] bool
    n_psd = int(mask.sum().item())

    if n_psd == 0:
        # 该批次没有低 t 样本, L_psd 设为 0 (保持 dtype/device)
        loss_psd = torch.zeros((), dtype=loss_diff.dtype, device=loss_diff.device)
    else:
        # 2. 全批次反推 x̂_0 (保留所有样本的梯度路径)
        x0_hat = schedule.predict_x0_from_eps(x_t, t, eps_pred)  # [B, 2, L]

        # 3. 估计 n̂ 的 PSD
        n_hat = y - x0_hat                                       # [B, 2, L]
        psd_pred_log = welch_psd_log(
            n_hat,
            nperseg=psd_nperseg,
            noverlap=psd_noverlap,
            fs=psd_fs,
            eps=psd_eps,
            detrend=psd_detrend,
        )  # [B, 256]

        # 4. 逐样本 MSE (在频率维上取均值, 留下 batch 维做 mask)
        per_sample_mse = (psd_pred_log - psd_gt_log).pow(2).mean(dim=-1)  # [B]

        # 5. 应用 mask: 只在 mask=True 的样本上累加, 取这些样本的均值
        mask_f = mask.to(per_sample_mse.dtype)                            # [B] float
        loss_psd = (per_sample_mse * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    # ===== 合并 =====
    loss_total = loss_diff + lambda_psd * loss_psd

    return LossOutput(
        total=loss_total,
        diff=loss_diff.detach(),
        psd=loss_psd.detach(),
        n_psd_samples=n_psd,
    )


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.diffusion.train_loss
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 准备测试 schedule
    sched = DiffusionSchedule(num_timesteps=1000, schedule_type="linear")

    B, C, L = 8, 2, 1024

    # 模拟数据: x_0 干净信号, n 噪声, y = x_0 + n
    x_0 = torch.randn(B, C, L) * 0.5
    n = torch.randn(B, C, L) * 0.5
    y = x_0 + n

    # 真实噪声的 log PSD
    psd_gt_log = welch_psd_log(n)  # [B, 256]
    print(f"psd_gt_log shape: {tuple(psd_gt_log.shape)}")

    # 模拟训练 step
    t = torch.randint(0, 1000, (B,))
    eps_target = torch.randn(B, C, L)
    x_t = sched.q_sample(x_0, t, eps_target)

    # 模拟网络预测 (带噪声偏差, 有可学习参数)
    eps_pred = eps_target + 0.1 * torch.randn(B, C, L)
    eps_pred.requires_grad_(True)

    print("\n" + "=" * 60)
    print("基础功能测试 (随机 t, 默认 threshold=250)")
    print("=" * 60)
    print(f"批次 t 值: {t.tolist()}")
    print(f"低 t (<250) 样本数: {(t < 250).sum().item()} / {B}")

    out = compute_pcddm_loss(
        eps_pred=eps_pred,
        eps_target=eps_target,
        x_t=x_t,
        y=y,
        t=t,
        schedule=sched,
        psd_gt_log=psd_gt_log,
        lambda_psd=0.1,
        psd_loss_threshold=250,
    )

    print(f"\nL_diff: {out.diff.item():.6f}")
    print(f"L_psd:  {out.psd.item():.6f}")
    print(f"L_total: {out.total.item():.6f}")
    print(f"  应满足 total = diff + 0.1 * psd = "
          f"{out.diff.item() + 0.1 * out.psd.item():.6f}")
    print(f"n_psd_samples: {out.n_psd_samples}")

    # ===== 反向传播测试 =====
    print("\n" + "=" * 60)
    print("反向传播测试")
    print("=" * 60)
    out.total.backward()
    print(f"eps_pred.grad shape: {tuple(eps_pred.grad.shape)}")
    print(f"eps_pred.grad norm : {eps_pred.grad.norm().item():.4f}")
    print(f"梯度非零: {'✓' if eps_pred.grad.abs().sum() > 0 else '✗'}")

    # ===== 边界情况: 全部 t 都 >= threshold =====
    print("\n" + "=" * 60)
    print("边界: 全部 t >= threshold (L_psd 应为 0)")
    print("=" * 60)
    eps_pred2 = (eps_target + 0.1 * torch.randn(B, C, L)).requires_grad_(True)
    t_high = torch.full((B,), 800, dtype=torch.long)
    x_t2 = sched.q_sample(x_0, t_high, eps_target)
    out2 = compute_pcddm_loss(
        eps_pred2, eps_target, x_t2, y, t_high, sched, psd_gt_log,
        lambda_psd=0.1, psd_loss_threshold=250,
    )
    print(f"L_diff: {out2.diff.item():.6f}")
    print(f"L_psd:  {out2.psd.item():.6f} (期望 0)")
    print(f"n_psd_samples: {out2.n_psd_samples} (期望 0)")
    out2.total.backward()
    print(f"高 t 时仍能反传: {'✓' if eps_pred2.grad.abs().sum() > 0 else '✗'}")

    # ===== 边界情况: 全部 t 都 < threshold =====
    print("\n" + "=" * 60)
    print("边界: 全部 t < threshold (L_psd 全样本累计)")
    print("=" * 60)
    eps_pred3 = (eps_target + 0.1 * torch.randn(B, C, L)).requires_grad_(True)
    t_low = torch.full((B,), 100, dtype=torch.long)
    x_t3 = sched.q_sample(x_0, t_low, eps_target)
    out3 = compute_pcddm_loss(
        eps_pred3, eps_target, x_t3, y, t_low, sched, psd_gt_log,
        lambda_psd=0.1, psd_loss_threshold=250,
    )
    print(f"L_diff: {out3.diff.item():.6f}")
    print(f"L_psd:  {out3.psd.item():.6f} (期望 > 0)")
    print(f"n_psd_samples: {out3.n_psd_samples} (期望 {B})")

    # ===== 完美预测 -> L_diff = 0, L_psd ≈ 0 =====
    print("\n" + "=" * 60)
    print("完美预测 (ε̂ = ε): L_diff = 0, L_psd 应也很小")
    print("=" * 60)
    eps_perfect = eps_target.clone().requires_grad_(True)
    t_low = torch.full((B,), 50, dtype=torch.long)
    x_t_perf = sched.q_sample(x_0, t_low, eps_target)
    # 注意: x_0_hat 用 ε̂=ε 反推应该完全恢复 x_0, 然后 n̂ = y - x_0 = n, psd 一致
    out_perf = compute_pcddm_loss(
        eps_perfect, eps_target, x_t_perf, y, t_low, sched, psd_gt_log,
        lambda_psd=0.1, psd_loss_threshold=250,
    )
    print(f"L_diff: {out_perf.diff.item():.6e} (期望 ~0)")
    print(f"L_psd:  {out_perf.psd.item():.6e} (期望 ~0, 因为 n̂ 应等于 n)")

    # ===== λ=0 时 L_psd 不影响 total =====
    print("\n" + "=" * 60)
    print("λ_psd=0 时 L_total = L_diff")
    print("=" * 60)
    eps_pred4 = (eps_target + 0.1 * torch.randn(B, C, L)).requires_grad_(True)
    out4 = compute_pcddm_loss(
        eps_pred4, eps_target, x_t, y, t, sched, psd_gt_log,
        lambda_psd=0.0, psd_loss_threshold=250,
    )
    print(f"L_diff: {out4.diff.item():.6f}")
    print(f"L_psd:  {out4.psd.item():.6f} (仍计算但不参与 total)")
    print(f"L_total: {out4.total.item():.6f}")
    diff_check = abs(out4.total.item() - out4.diff.item()) < 1e-6
    print(f"  L_total == L_diff: {'✓' if diff_check else '✗'}")

    # ===== GPU 测试 =====
    if torch.cuda.is_available():
        sched_gpu = sched.cuda()
        eps_pred_gpu = eps_pred.detach().cuda().requires_grad_(True)
        out_gpu = compute_pcddm_loss(
            eps_pred_gpu, eps_target.cuda(), x_t.cuda(), y.cuda(),
            t.cuda(), sched_gpu, psd_gt_log.cuda(),
            lambda_psd=0.1, psd_loss_threshold=250,
        )
        print(f"\n[GPU] L_total: {out_gpu.total.item():.6f}, device: {out_gpu.total.device}")
