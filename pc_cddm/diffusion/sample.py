"""
PC-CDDM 反向采样推理链路。

实现两种采样策略:
    DDPM: 标准随机反向扩散, 全 T 步, 每步加高斯噪声 (η=1)
    DDIM: 确定性加速采样 (η=0), 在 [0, T-1] 上线性等距取 num_inference_steps 步

两者共同特性:
    - 每 psd_refine_interval 步用当前 x̂_0 反推 n̂ = y - x̂_0, 重估 psd_hat
      (低 t 时 x̂_0 越来越可信, 重估越来越有意义)
    - 初始 psd_hat 从 y 直接估计 (低 SNR 时 y 谱 ≈ n 谱)

推理核心公式:
    DDPM 单步:
        ε̂ = UNet(x_t, c)
        x̂_0 = predict_x0_from_eps(x_t, t, ε̂)
        μ   = q_posterior_mean(x̂_0, x_t, t)
        x_{t-1} = μ + σ_t · z   (t > 0),  z ~ N(0, I)
                = μ              (t = 0)
        σ_t = √posterior_variance[t]

    DDIM 单步 (η=0, 确定性):
        ε̂ = UNet(x_τ, c)
        x̂_0 = predict_x0_from_eps(x_τ, τ, ε̂)
        x_{τ_prev} = √ᾱ_{τ_prev} · x̂_0 + √(1 - ᾱ_{τ_prev}) · ε̂

关键设计红线:
    评估 NMSE 必须走完整反向采样链得到 x̂_0 后计算, 不可用训练随机 t 的 loss 替代。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from pc_cddm.utils.psd import welch_psd_log
from pc_cddm.diffusion.schedule import DiffusionSchedule


# ============================================================================
# 工具: 构造 DDIM 时间子序列
# ============================================================================
def _make_ddim_timesteps(T: int, num_inference_steps: int) -> torch.Tensor:
    """
    在 [0, T-1] 上线性等距取 num_inference_steps 个整数时间步, 降序排列。

    例: T=1000, steps=20 -> [999, 947, 894, ..., 52, 0] (共 20 个)

    Args:
        T: 总时间步数
        num_inference_steps: DDIM 推理步数

    Returns:
        timesteps: [num_inference_steps] long, 降序
    """
    # 在 [0, T-1] 等距取点 (含两端), 四舍五入取整
    ts = torch.linspace(0, T - 1, num_inference_steps).round().long()  # [S] 升序
    return ts.flip(0)  # 降序, 从 t=T-1 开始


# ============================================================================
# DDPM 单步反向
# ============================================================================
def _ddpm_step(
    x_t: torch.Tensor,           # [B, 2, L]
    t_idx: int,
    eps_pred: torch.Tensor,      # [B, 2, L]
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DDPM 反向单步: x_t -> x_{t-1}。

    Returns:
        x_prev: [B, 2, L]  采样结果 x_{t-1}
        x0_hat: [B, 2, L]  当前步反推的 x̂_0 (供 PSD 重估使用)
    """
    B = x_t.size(0)
    device = x_t.device
    t_batch = torch.full((B,), t_idx, dtype=torch.long, device=device)  # [B]

    # x̂_0 反推
    x0_hat = schedule.predict_x0_from_eps(x_t, t_batch, eps_pred)  # [B, 2, L]

    # 后验均值 μ
    mu = schedule.q_posterior_mean(x0_hat, x_t, t_batch)  # [B, 2, L]

    # 加噪 (t > 0 时加, t = 0 时直接返回 μ)
    if t_idx > 0:
        # σ_t = √posterior_variance[t]
        sigma = schedule.posterior_variance[t_idx].sqrt()   # 标量
        z = torch.randn_like(x_t)                           # [B, 2, L]
        x_prev = mu + sigma * z
    else:
        x_prev = mu

    return x_prev, x0_hat


# ============================================================================
# DDIM 单步反向 (η=0, 确定性)
# ============================================================================
def _ddim_step(
    x_tau: torch.Tensor,          # [B, 2, L] 当前时刻 x_τ
    tau: int,                     # 当前时间步索引
    tau_prev: int,                # 上一时间步索引
    eps_pred: torch.Tensor,       # [B, 2, L]
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DDIM 确定性单步: x_τ -> x_{τ_prev}。

    x_{τ_prev} = √ᾱ_{τ_prev} · x̂_0 + √(1 - ᾱ_{τ_prev}) · ε̂

    Returns:
        x_prev: [B, 2, L]
        x0_hat: [B, 2, L]
    """
    B = x_tau.size(0)
    device = x_tau.device
    t_batch = torch.full((B,), tau, dtype=torch.long, device=device)  # [B]

    # x̂_0 反推
    x0_hat = schedule.predict_x0_from_eps(x_tau, t_batch, eps_pred)  # [B, 2, L]

    # ᾱ_{τ_prev}: tau_prev=-1 时约定 ᾱ=1 (对应 x̂_0 本身, 不会实际发生)
    alpha_bar_prev = schedule.alphas_cumprod[tau_prev]   # 标量

    # DDIM 确定性更新
    # x_{τ_prev} = √ᾱ_{τ_prev} · x̂_0 + √(1-ᾱ_{τ_prev}) · ε̂
    x_prev = (
        alpha_bar_prev.sqrt() * x0_hat
        + (1.0 - alpha_bar_prev).sqrt() * eps_pred
    )  # [B, 2, L]

    return x_prev, x0_hat


# ============================================================================
# 主采样函数
# ============================================================================
@torch.no_grad()
def sample(
    unet: nn.Module,
    condition_encoder: nn.Module,
    schedule: DiffusionSchedule,
    y: torch.Tensor,                      # [B, 2, L] 含噪观测
    snr_db: torch.Tensor,                 # [B] 物理 SNR (dB)
    *,
    method: str = "ddpm",                 # "ddpm" | "ddim"
    num_inference_steps: Optional[int] = None,  # None 时 DDPM 用全 T 步
    psd_refine_interval: int = 50,        # 每隔多少步重估 psd_hat
    psd_init: str = "from_y",            # "from_y" (目前唯一支持)
    psd_nperseg: int = 256,
    psd_noverlap: int = 128,
    psd_eps: float = 1e-8,
    progress: bool = False,
) -> torch.Tensor:                        # [B, 2, L] 去噪后 x̂_0
    """
    PC-CDDM 完整反向采样推理。

    Args:
        unet:              训练好的 UNet1D 模型
        condition_encoder: ConditionEncoder 模型
        schedule:          DiffusionSchedule (含所有预计算 buffer)
        y:                 [B, 2, L] 含噪 IQ 观测信号
        snr_db:            [B] 物理 SNR 标量 (dB, 与训练一致)
        method:            "ddpm" 随机全链路 | "ddim" 确定性加速
        num_inference_steps: DDIM 步数; DDPM 时默认 schedule.T
        psd_refine_interval: 每隔多少采样步重估 psd_hat (默认 50)
        psd_init:          初始 psd_hat 来源, 目前仅支持 "from_y"
        psd_nperseg:       Welch nperseg, 须与训练一致 (默认 256)
        psd_noverlap:      Welch noverlap (默认 128)
        psd_eps:           log PSD 数值保护
        progress:          是否显示 tqdm 进度条

    Returns:
        x0_hat: [B, 2, L] 反向链末端输出的去噪信号
    """
    # ------------------------------------------------------------------
    # 基础校验
    # ------------------------------------------------------------------
    if method not in ("ddpm", "ddim"):
        raise ValueError(f"method 期望 'ddpm' | 'ddim', 实际 '{method}'")

    device = y.device
    B, C, L = y.shape   # [B, 2, 1024]
    T = schedule.T

    # ------------------------------------------------------------------
    # 构造时间步序列
    # ------------------------------------------------------------------
    if method == "ddpm":
        # 全链路: [T-1, T-2, ..., 1, 0]
        timesteps = torch.arange(T - 1, -1, -1, dtype=torch.long)  # [T]
        # DDPM 不需要 "上一步" 索引, 由单步函数内部处理
        prev_timesteps = None
    else:
        # DDIM: 线性子序列, 降序
        S = num_inference_steps if num_inference_steps is not None else T
        timesteps = _make_ddim_timesteps(T, S)  # [S] 降序
        # τ_prev: 右移一位, 最后一步 tau_prev=0 (子序列最小值)
        prev_timesteps = torch.cat([
            timesteps[1:],
            torch.zeros(1, dtype=torch.long)
        ])  # [S]

    total_steps = len(timesteps)

    # ------------------------------------------------------------------
    # 进度条 (可选)
    # ------------------------------------------------------------------
    try:
        from tqdm.auto import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    def _iter(seq):
        if progress and _tqdm_available:
            return tqdm(seq, total=total_steps, desc=f"sampling [{method}]")
        return seq

    # ------------------------------------------------------------------
    # 初始化 x_T ~ N(0, I)
    # ------------------------------------------------------------------
    x_t = torch.randn(B, C, L, device=device)  # [B, 2, 1024]

    # ------------------------------------------------------------------
    # 初始 psd_hat: 低 SNR 时 y 谱近似为 n 谱
    # ------------------------------------------------------------------
    if psd_init == "from_y":
        psd_hat = welch_psd_log(
            y,
            nperseg=psd_nperseg,
            noverlap=psd_noverlap,
            eps=psd_eps,
        )  # [B, 256]
    else:
        raise ValueError(f"psd_init 目前仅支持 'from_y', 实际 '{psd_init}'")

    # ------------------------------------------------------------------
    # 切换模型到 eval 模式 (保存原始状态, 结束后恢复)
    # ------------------------------------------------------------------
    unet_training = unet.training
    ce_training = condition_encoder.training
    unet.eval()
    condition_encoder.eval()

    psd_refine_count = 0   # 记录实际重估次数 (供自检验证)

    # ------------------------------------------------------------------
    # 主反向采样循环
    # ------------------------------------------------------------------
    for step_i, t_idx in enumerate(_iter(range(total_steps))):
        tau = int(timesteps[step_i])  # 当前时间步整数值

        # ---- 构建条件向量 c ----
        t_batch = torch.full((B,), tau, dtype=torch.long, device=device)       # [B]
        c = condition_encoder(t_batch, snr_db, psd_hat)  # [B, 256]

        # ---- UNet 预测 ε̂ ----
        eps_pred = unet(x_t, c)  # [B, 2, 1024]

        # ---- 反向单步 ----
        if method == "ddpm":
            x_t, x0_hat = _ddpm_step(x_t, tau, eps_pred, schedule)
        else:
            tau_prev = int(prev_timesteps[step_i])
            x_t, x0_hat = _ddim_step(x_t, tau, tau_prev, eps_pred, schedule)

        # ---- PSD 重估 (每隔 K 步, 用最新 x̂_0 反推 n̂ = y - x̂_0) ----
        # step_i+1: 让第 0 步刚走完后也能触发第一次重估 (若 interval=1)
        if (step_i + 1) % psd_refine_interval == 0:
            n_hat = y - x0_hat                    # [B, 2, 1024] 估计噪声
            psd_hat = welch_psd_log(
                n_hat,
                nperseg=psd_nperseg,
                noverlap=psd_noverlap,
                eps=psd_eps,
            )  # [B, 256]
            psd_refine_count += 1

    # ------------------------------------------------------------------
    # 恢复模型训练状态
    # ------------------------------------------------------------------
    unet.train(unet_training)
    condition_encoder.train(ce_training)

    # x_t 已走到 t=0, 此时即为 x̂_0
    x0_hat = x_t  # [B, 2, 1024]

    # 附带 refine 次数便于外部验证 (非正式接口, 不进入 type signature)
    x0_hat._psd_refine_count = psd_refine_count  # type: ignore[attr-defined]

    return x0_hat


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.diffusion.sample
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    torch.manual_seed(0)

    # ---- 导入所有依赖模块 ----
    from pc_cddm.diffusion.schedule import DiffusionSchedule
    from pc_cddm.models.unet1d import UNet1D
    from pc_cddm.models.condition_encoder import ConditionEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 构造模型 (未训练权重, 用于 shape / NaN 验证) ----
    fake_model_cfg = {
        "time_embed_dim": 128,
        "time_mlp_dim": 64,
        "snr_embed_dim": 128,
        "snr_mlp_dim": 64,
        "psd_mlp_hidden": 256,
        "psd_mlp_out": 64,
        "cond_dim": 256,
        "base_channels": 64,
        "channel_mults": [1, 2, 4],
        "num_res_blocks": 2,
        "groupnorm_groups": 8,
        "snr_min": -15.0,
        "snr_max": 10.0,
    }
    fake_psd_cfg = {
        "nperseg": 256,
        "noverlap": 128,
        "fs": 1.0,
        "eps": 1e-8,
        "detrend": True,
    }
    fake_diff_cfg = {
        "num_timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }

    unet = UNet1D.from_config(fake_model_cfg).to(device)
    ce = ConditionEncoder.from_config(fake_model_cfg, fake_psd_cfg).to(device)
    schedule = DiffusionSchedule.from_config(fake_diff_cfg).to(device)

    # 模拟训练一步赋权重 (否则 GroupNorm + FiLM 恒等初始化输出全零会触发边界)
    # 只做一次随机权重扰动即可, 无需真实训练
    with torch.no_grad():
        for p in unet.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)
        for p in ce.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)

    B, C, L = 2, 2, 1024
    y = torch.randn(B, C, L, device=device)          # [B, 2, 1024] 含噪观测
    snr_db = torch.tensor([-5.0, 0.0], device=device)  # [B]

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 1: DDPM 短链路 (50 步)")
    print("=" * 60)
    # ==================================================================
    x0_ddpm = sample(
        unet, ce, schedule,
        y, snr_db,
        method="ddpm",
        num_inference_steps=50,   # DDPM 模式下此参数被忽略, 实际走全 T=1000 步
        psd_refine_interval=50,
        progress=False,
    )
    # 注: DDPM 走全 T=1000 步太慢, 改成缩短 T 的 schedule 做自检
    # 实际用更小的 T 做快速 shape 检查
    fast_diff_cfg = {**fake_diff_cfg, "num_timesteps": 50}
    schedule_fast = DiffusionSchedule.from_config(fast_diff_cfg).to(device)

    x0_ddpm = sample(
        unet, ce, schedule_fast,
        y, snr_db,
        method="ddpm",
        psd_refine_interval=10,
        progress=False,
    )
    refine_count_ddpm = getattr(x0_ddpm, "_psd_refine_count", -1)
    expected_refine_ddpm = 50 // 10   # total_steps=50, interval=10 -> 5 次

    print(f"输出 shape: {tuple(x0_ddpm.shape)}  (期望 [{B}, 2, 1024])")
    print(f"shape 正确: {'✓' if x0_ddpm.shape == (B, C, L) else '✗'}")
    print(f"NaN 检查:   {'✓ 无 NaN' if not x0_ddpm.isnan().any() else '✗ 有 NaN'}")
    print(f"Inf 检查:   {'✓ 无 Inf' if not x0_ddpm.isinf().any() else '✗ 有 Inf'}")
    print(f"设备一致:   {'✓' if x0_ddpm.device == device else '✗'}")
    print(f"PSD 重估次数: {refine_count_ddpm}  (期望 {expected_refine_ddpm})")
    print(f"PSD 重估正确: {'✓' if refine_count_ddpm == expected_refine_ddpm else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: DDIM 短链路 (20 步)")
    print("=" * 60)
    # ==================================================================
    x0_ddim = sample(
        unet, ce, schedule,                # 用完整 T=1000 schedule
        y, snr_db,
        method="ddim",
        num_inference_steps=20,
        psd_refine_interval=5,
        progress=False,
    )
    refine_count_ddim = getattr(x0_ddim, "_psd_refine_count", -1)
    expected_refine_ddim = 20 // 5   # total_steps=20, interval=5 -> 4 次

    print(f"输出 shape: {tuple(x0_ddim.shape)}  (期望 [{B}, 2, 1024])")
    print(f"shape 正确: {'✓' if x0_ddim.shape == (B, C, L) else '✗'}")
    print(f"NaN 检查:   {'✓ 无 NaN' if not x0_ddim.isnan().any() else '✗ 有 NaN'}")
    print(f"Inf 检查:   {'✓ 无 Inf' if not x0_ddim.isinf().any() else '✗ 有 Inf'}")
    print(f"设备一致:   {'✓' if x0_ddim.device == device else '✗'}")
    print(f"PSD 重估次数: {refine_count_ddim}  (期望 {expected_refine_ddim})")
    print(f"PSD 重估正确: {'✓' if refine_count_ddim == expected_refine_ddim else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: DDIM 时间子序列单调性")
    print("=" * 60)
    # ==================================================================
    ts = _make_ddim_timesteps(1000, 20)
    print(f"DDIM 子序列 (T=1000, 20 步): {ts.tolist()}")
    print(f"降序: {'✓' if (ts[:-1] > ts[1:]).all() else '✗'}")
    print(f"最大值 ≤ T-1: {'✓' if ts.max() <= 999 else '✗'}")
    print(f"最小值 ≥ 0:   {'✓' if ts.min() >= 0 else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: eval/train 状态恢复")
    print("=" * 60)
    # ==================================================================
    unet.train()
    ce.train()
    _ = sample(unet, ce, schedule_fast, y, snr_db, method="ddpm",
               psd_refine_interval=50, progress=False)
    print(f"unet.training 采样后恢复 True:  {'✓' if unet.training else '✗'}")
    print(f"ce.training   采样后恢复 True:  {'✓' if ce.training else '✗'}")

    unet.eval()
    ce.eval()
    _ = sample(unet, ce, schedule_fast, y, snr_db, method="ddpm",
               psd_refine_interval=50, progress=False)
    print(f"unet.training 采样后恢复 False: {'✓' if not unet.training else '✗'}")
    print(f"ce.training   采样后恢复 False: {'✓' if not ce.training else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 5: 无效 method 参数报错")
    print("=" * 60)
    # ==================================================================
    try:
        _ = sample(unet, ce, schedule_fast, y, snr_db, method="euler")
        print("✗ 应当抛出 ValueError 但未抛出")
    except ValueError as e:
        print(f"✓ 正确抛出 ValueError: {e}")

    print("\n" + "=" * 60)
    print("全部自检完成")
    print("=" * 60)
