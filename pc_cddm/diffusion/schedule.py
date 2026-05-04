"""
DiffusionSchedule: β 调度与扩散过程数学工具。

包含:
    1. β 调度: linear (Ho 2020) / cosine (Nichol & Dhariwal 2021)
    2. 所有派生量预计算并注册为 buffer (跟着 .to(device) 走)
    3. q_sample: 前向加噪 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    4. predict_x0_from_eps: 从预测噪声反推 x̂_0 (用于 L_psd 和推理 PSD 重估)

数学约定 (DDPM, Ho et al. 2020):
    β_t ∈ (0, 1):           扩散方差调度
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^t α_s     累积乘积
    q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) I)
    posterior σ²_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)   (反向采样方差)

数值稳定性:
    - cosine schedule 的 β 钳到 ≤ 0.999, 避免 α_t -> 0
    - posterior_variance[0] = β[0] (避免 t=0 处 ᾱ_{t-1} 未定义)
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn


# ============================================================================
# β 调度构造
# ============================================================================
def _linear_beta_schedule(
    T: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """
    Ho et al. 2020 标准 linear schedule。
    β_t 在 [beta_start, beta_end] 上线性等距, 长度 T。

    Returns:
        betas: [T]
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal 2021 cosine schedule。
    ᾱ_t = cos²((t/T + s) / (1+s) · π/2), 然后反推 β。

    Args:
        T: 时间步数
        s: 偏移量, 默认 0.008 (论文推荐)

    Returns:
        betas: [T], 钳到 ≤ 0.999
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T  # [T+1]
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2  # [T+1]
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]               # 归一到 ᾱ_0 = 1
    # β_t = 1 - ᾱ_t / ᾱ_{t-1}
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]            # [T]
    return betas.clamp(max=0.999)


def make_beta_schedule(
    schedule_type: str,
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """
    构造 β 调度。

    Args:
        schedule_type: "linear" | "cosine"
        T: 时间步数
        beta_start / beta_end: linear 用
        cosine_s: cosine 用

    Returns:
        betas: [T]
    """
    if schedule_type == "linear":
        return _linear_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "cosine":
        return _cosine_beta_schedule(T, cosine_s)
    else:
        raise ValueError(f"未知 schedule_type: {schedule_type}, 期望 linear|cosine")


# ============================================================================
# 工具: 按 t 索引并 reshape 成可广播 shape
# ============================================================================
def extract(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    从一维数组 arr [T] 按时间步索引, reshape 为 [B, 1, 1, ...] 形式以便广播到 x_shape。

    Args:
        arr:     [T] 调度数组 (e.g. sqrt_alphas_cumprod)
        t:       [B] long 时间步索引
        x_shape: 目标张量形状, e.g. (B, 2, 1024)

    Returns:
        [B, 1, 1, ...] 与 x_shape 维度数相同, 除 batch 外其余维度为 1
    """
    B = t.size(0)
    out = arr.gather(0, t)  # [B]
    # reshape 成 [B, 1, 1, ...]
    return out.reshape(B, *((1,) * (len(x_shape) - 1)))


# ============================================================================
# DiffusionSchedule 主类
# ============================================================================
class DiffusionSchedule(nn.Module):
    """
    扩散调度容器: 持有所有预计算的 schedule buffer, 提供 q_sample 等方法。

    继承 nn.Module 让 buffer 自动跟着 .to(device) 走。

    Args:
        num_timesteps:  T
        schedule_type:  "linear" | "cosine"
        beta_start:     linear 起始 β
        beta_end:       linear 终止 β
        cosine_s:       cosine 偏移
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
    ):
        super().__init__()
        self.T = num_timesteps
        self.schedule_type = schedule_type

        # 构造 β (在 float64 下计算, 最后转 float32 存)
        betas = make_beta_schedule(
            schedule_type, num_timesteps, beta_start, beta_end, cosine_s
        )  # [T] float64

        # 派生量 (全部 float64 计算, 避免累积乘积精度损失)
        alphas = 1.0 - betas                              # [T]
        alphas_cumprod = torch.cumprod(alphas, dim=0)     # [T] ᾱ_t

        # ᾱ_{t-1}, t=0 时取 1.0
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=torch.float64),
            alphas_cumprod[:-1]
        ])  # [T]

        # 前向加噪用
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)               # [T]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # [T]

        # 反向采样用 (DDPM 公式: x_{t-1} 的均值需要)
        # x_0 反推: x_0 = (x_t - √(1-ᾱt) · ε) / √ᾱt
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)        # [T]
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)  # [T]

        # Posterior σ²_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # t=0 时 (1-ᾱ_t) 可能很小但非零 (因为 ᾱ_0 ≈ 1-β_0); 这里手动设 posterior[0] = β[0]
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # t=0 校正: 此时分母 (1-ᾱ_0) = β_0 极小, 公式仍数值稳定, 但为保险计:
        posterior_variance[0] = betas[0]
        # log clamp 用于反向采样的 log variance (避免 log 0)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        # 反向采样均值系数 (Ho 2020 Eq. 7, μ_θ = coef1 · x_0 + coef2 · x_t)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # 全部转 float32 注册为 buffer
        def reg(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.to(torch.float32))

        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        reg("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        reg("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        reg("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", posterior_log_variance_clipped)
        reg("posterior_mean_coef1", posterior_mean_coef1)
        reg("posterior_mean_coef2", posterior_mean_coef2)

    # ------------------------------------------------------------------
    # 前向加噪 (训练用)
    # ------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        从干净信号 x_0 直接采样 t 时刻的加噪样本 x_t (跳过中间步)。

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)

        Args:
            x_0:   [B, C, L] 干净信号
            t:     [B] long 时间步
            noise: [B, C, L] 噪声, 默认从 N(0,I) 采样

        Returns:
            x_t: [B, C, L]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_0.shape)            # [B,1,1]
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)  # [B,1,1]

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # 从 ε̂ 反推 x̂_0 (PSD 损失 / 推理 PSD 重估用)
    # ------------------------------------------------------------------
    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        x̂_0 = (x_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t
             = √(1/ᾱ_t) · x_t - √(1/ᾱ_t - 1) · ε̂

        Args:
            x_t: [B, C, L] 加噪信号
            t:   [B] long 时间步
            eps: [B, C, L] 网络预测的噪声 ε̂

        Returns:
            x_0_hat: [B, C, L]
        """
        coef1 = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * eps

    # ------------------------------------------------------------------
    # 反向采样均值 (sample.py 会用)
    # ------------------------------------------------------------------
    def q_posterior_mean(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 q(x_{t-1} | x_t, x_0) 的均值。

        μ = coef1(t) · x_0 + coef2(t) · x_t

        Args:
            x_0: [B, C, L]
            x_t: [B, C, L]
            t:   [B] long

        Returns:
            μ: [B, C, L]
        """
        coef1 = extract(self.posterior_mean_coef1, t, x_0.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_0 + coef2 * x_t

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, diff_cfg: dict[str, Any]) -> "DiffusionSchedule":
        """
        从 yaml['diffusion'] 段构造。
        """
        return cls(
            num_timesteps=diff_cfg["num_timesteps"],
            schedule_type=diff_cfg["beta_schedule"],
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
            cosine_s=diff_cfg.get("cosine_s", 0.008),
        )


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.diffusion.schedule
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # ===== Linear schedule =====
    print("=" * 60)
    print("Linear schedule (T=1000, β: 1e-4 -> 0.02)")
    print("=" * 60)
    sched = DiffusionSchedule(
        num_timesteps=1000,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    print(f"betas: shape={tuple(sched.betas.shape)}, "
          f"min={sched.betas.min():.2e}, max={sched.betas.max():.2e}")
    print(f"alphas_cumprod: ᾱ_0 = {sched.alphas_cumprod[0]:.6f} (期望 ≈ 1-β_0 ≈ 0.9999)")
    print(f"                ᾱ_T = {sched.alphas_cumprod[-1]:.6e} (期望 ≈ 0)")
    print(f"sqrt_alphas_cumprod[0]   = {sched.sqrt_alphas_cumprod[0]:.6f}")
    print(f"sqrt_alphas_cumprod[T-1] = {sched.sqrt_alphas_cumprod[-1]:.6e}")

    # ===== Cosine schedule =====
    print("\n" + "=" * 60)
    print("Cosine schedule (T=1000, s=0.008)")
    print("=" * 60)
    sched_c = DiffusionSchedule(
        num_timesteps=1000,
        schedule_type="cosine",
        cosine_s=0.008,
    )
    print(f"betas: shape={tuple(sched_c.betas.shape)}, "
          f"min={sched_c.betas.min():.2e}, max={sched_c.betas.max():.4f}")
    print(f"  cosine 末端 β 钳到 ≤ 0.999: {'✓' if sched_c.betas.max() <= 0.999 + 1e-6 else '✗'}")
    print(f"alphas_cumprod: ᾱ_0 = {sched_c.alphas_cumprod[0]:.6f}")
    print(f"                ᾱ_T = {sched_c.alphas_cumprod[-1]:.6e}")

    # ===== q_sample 数学验证 =====
    print("\n" + "=" * 60)
    print("q_sample 数学验证")
    print("=" * 60)
    B, C, L = 4, 2, 1024
    x_0 = torch.randn(B, C, L)
    noise = torch.randn(B, C, L)

    # t=0: x_t 应几乎等于 x_0 (因为 ᾱ_0 ≈ 1, √(1-ᾱ_0) ≈ 0)
    t0 = torch.zeros(B, dtype=torch.long)
    x_t0 = sched.q_sample(x_0, t0, noise)
    diff_t0 = (x_t0 - x_0).abs().mean().item()
    print(f"t=0: mean|x_t - x_0| = {diff_t0:.4e}  (期望接近 0)")

    # t=T-1: x_t 应几乎等于 noise (ᾱ_T ≈ 0, √(1-ᾱ_T) ≈ 1)
    tT = torch.full((B,), 999, dtype=torch.long)
    x_tT = sched.q_sample(x_0, tT, noise)
    diff_tT = (x_tT - noise).abs().mean().item()
    print(f"t=T-1: mean|x_t - ε| = {diff_tT:.4e}  (期望接近 0)")

    # 中间 t: 验证方差
    # x_t 的方差应等于 ᾱ_t · Var(x_0) + (1-ᾱ_t) · Var(ε) = ᾱ_t + (1-ᾱ_t) = 1
    # (假设 x_0 和 ε 都是单位方差, 这里用 randn 满足)
    t_mid = torch.full((B,), 500, dtype=torch.long)
    # 用大批次降低方差估计抖动
    x_0_big = torch.randn(256, C, L)
    eps_big = torch.randn(256, C, L)
    t_mid_big = torch.full((256,), 500, dtype=torch.long)
    x_t_big = sched.q_sample(x_0_big, t_mid_big, eps_big)
    var_xt = x_t_big.var().item()
    print(f"t=500: Var(x_t) = {var_xt:.4f}  (期望 ≈ 1.0)")

    # ===== predict_x0_from_eps 往返一致性 =====
    print("\n" + "=" * 60)
    print("predict_x0_from_eps 往返一致性")
    print("=" * 60)
    # x_0 -> x_t -> 用 noise 反推 -> 应得回 x_0
    for t_val in [0, 100, 500, 999]:
        t = torch.full((B,), t_val, dtype=torch.long)
        x_t = sched.q_sample(x_0, t, noise)
        x_0_recovered = sched.predict_x0_from_eps(x_t, t, noise)
        err = (x_0_recovered - x_0).abs().max().item()
        print(f"t={t_val:>4d}: max|x_0_recovered - x_0| = {err:.4e}")

    # ===== posterior_variance 合理性 =====
    print("\n" + "=" * 60)
    print("Posterior variance 合理性")
    print("=" * 60)
    print(f"σ²_0 = {sched.posterior_variance[0]:.6e} (设为 β_0)")
    print(f"σ²_T = {sched.posterior_variance[-1]:.6e}")
    print(f"σ²_500 = {sched.posterior_variance[500]:.6e}")
    print(f"全部非负: {'✓' if (sched.posterior_variance >= 0).all() else '✗'}")
    print(f"全部有限: {'✓' if torch.isfinite(sched.posterior_variance).all() else '✗'}")

    # ===== Buffer 跟着 .to(device) =====
    if torch.cuda.is_available():
        sched_gpu = sched.cuda()
        print(f"\n[GPU] sched.betas device: {sched_gpu.betas.device}")
        x_0_gpu = x_0.cuda()
        noise_gpu = noise.cuda()
        t_gpu = t0.cuda()
        x_t_gpu = sched_gpu.q_sample(x_0_gpu, t_gpu, noise_gpu)
        print(f"[GPU] q_sample 输出 device: {x_t_gpu.device}")

    # ===== from_config =====
    print("\nfrom_config 测试:")
    fake_cfg = {
        "num_timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }
    sched2 = DiffusionSchedule.from_config(fake_cfg)
    print(f"  构造成功, T = {sched2.T}, schedule = {sched2.schedule_type}")

    # ===== Linear vs Cosine 对比图 (不画图, 打印关键节点) =====
    print("\n" + "=" * 60)
    print("Linear vs Cosine ᾱ 对比 (关键节点)")
    print("=" * 60)
    for t_val in [0, 100, 250, 500, 750, 999]:
        a_lin = sched.alphas_cumprod[t_val].item()
        a_cos = sched_c.alphas_cumprod[t_val].item()
        print(f"  t={t_val:>4d}: linear ᾱ={a_lin:.4e}, cosine ᾱ={a_cos:.4e}")
