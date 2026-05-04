"""
Sinusoidal embeddings for diffusion timestep and SNR.

两个 embedding 设计差异点：
1. TimestepEmbedding: 标准 transformer 风格，频率指数跨 7~8 个数量级
                     (1.0 -> 1e-4)，适配 t ∈ [0, T-1] 的整数动态范围。
2. SNREmbedding:     密集低频设计，频率线性等距 [π/2, 16π]，适配
                     归一化到 [-1, 1] 的 SNR 标量，保持相邻 SNR 值
                     embedding 的平滑性。

每个模块都内部完成 "raw input -> sinusoidal expand -> 2-layer MLP" 全流程，
输出统一为 [B, mlp_dim] 维向量，供下游 condition_encoder 融合。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _sinusoidal_embedding(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    通用 sinusoidal expansion: 把标量 x 用一组频率展开成 [sin, cos] 拼接向量。

    Args:
        x:     [B] 标量输入
        freqs: [D/2] 频率参数 (buffer, 不参与训练)

    Returns:
        emb:   [B, D] 其中 D = 2 * len(freqs)
               emb[:, 0::2] = sin(x · freqs)
               emb[:, 1::2] = cos(x · freqs)
    """
    # x: [B] -> [B, 1]; freqs: [D/2] -> [1, D/2]
    # 广播相乘: [B, 1] * [1, D/2] -> [B, D/2]
    args = x.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, D/2]

    # 拼接 sin/cos: [B, D/2] + [B, D/2] -> [B, D]
    # 注意是 sin/cos 交替, 不是 [sin..., cos...] 块拼接
    sin_part = torch.sin(args)  # [B, D/2]
    cos_part = torch.cos(args)  # [B, D/2]
    # stack 到新维度然后 flatten 实现交替: [B, D/2, 2] -> [B, D]
    emb = torch.stack([sin_part, cos_part], dim=-1).flatten(start_dim=-2)  # [B, D]
    return emb


class TimestepEmbedding(nn.Module):
    """
    扩散时间步 embedding。

    流程:
        t (int) --[sinusoidal expand]--> [B, embed_dim]
                --[Linear -> SiLU -> Linear]--> [B, mlp_dim]

    频率设计 (标准 transformer 风格):
        freqs[i] = 1 / 10000^(2i / D),  i = 0..D/2-1
        范围: 1.0 (慢变) -> 1e-4 (快变), 跨 4 个数量级
        适配 t ∈ [0, T-1] 的整数动态范围
    """

    def __init__(self, embed_dim: int = 128, mlp_dim: int = 64):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必须为偶数，实际 {embed_dim}")

        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

        # 标准 transformer 频率: freqs[i] = 1 / 10000^(2i/D)
        # 对数空间下等距 -> 频率对数等距 (指数衰减)
        half = embed_dim // 2
        # log freqs: 0 -> -log(10000) 等距 (half 个点)
        # exp 后: 1 -> 1e-4 指数衰减
        log_max_period = math.log(10000.0)
        freqs = torch.exp(
            -log_max_period * torch.arange(half, dtype=torch.float32) / half
        )  # [half]
        self.register_buffer("freqs", freqs)  # [embed_dim // 2]

        # 2 层 MLP: embed_dim -> mlp_dim -> mlp_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 整数时间步 (long 或 float 都可，内部转 float)

        Returns:
            [B, mlp_dim]
        """
        if t.dim() != 1:
            raise ValueError(f"期望 t shape [B]，实际 {tuple(t.shape)}")

        t_float = t.to(self.freqs.dtype)  # [B]
        emb = _sinusoidal_embedding(t_float, self.freqs)  # [B, embed_dim]
        out = self.mlp(emb)  # [B, mlp_dim]
        return out


class SNREmbedding(nn.Module):
    """
    SNR (信噪比) embedding。

    流程:
        snr_db (float, 物理 dB) --[归一化到 [-1,1]]--> snr_norm
                                --[sinusoidal expand, 密集低频]--> [B, embed_dim]
                                --[Linear -> SiLU -> Linear]--> [B, mlp_dim]

    频率设计 (单独设计的密集低频):
        freqs = linspace(π/2, 4π, D/2)
        范围: 半周期 (π/2) -> 2 个完整周期 (4π) 在 [-1, 1] 上
        最低频维度: 编码 SNR 粗粒度 (高/低)
        最高频维度: 编码 SNR 精细差异 (区分 -5dB vs -4.5dB)
        注: 频率上限 4π 经验证可在 SNR 邻域 (差异 ≤ 5dB) 内保持距离
            严格单调; 远距离不单调是 sinusoidal embedding 的本质特性,
            不影响下游网络通过 embedding 向量本身做条件判别。

    归一化: s_norm = (snr_db - center) / half_range
        其中 center = (snr_min + snr_max) / 2
             half_range = (snr_max - snr_min) / 2
        默认 snr_min=-15, snr_max=10 -> center=-2.5, half_range=12.5
    """

    def __init__(
        self,
        embed_dim: int = 128,
        mlp_dim: int = 64,
        snr_min: float = -15.0,
        snr_max: float = 10.0,
        freq_low: float = math.pi / 2,
        freq_high: float = 4.0 * math.pi,
    ):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim 必须为偶数，实际 {embed_dim}")
        if snr_max <= snr_min:
            raise ValueError(f"snr_max ({snr_max}) 必须大于 snr_min ({snr_min})")

        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

        # SNR 归一化参数 (注册为 buffer，跟着模块迁 device)
        center = (snr_min + snr_max) / 2.0
        half_range = (snr_max - snr_min) / 2.0
        self.register_buffer("snr_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("snr_half_range", torch.tensor(half_range, dtype=torch.float32))

        # 密集低频: 线性等距 [freq_low, freq_high]
        half = embed_dim // 2
        freqs = torch.linspace(freq_low, freq_high, half, dtype=torch.float32)  # [half]
        self.register_buffer("freqs", freqs)

        # 2 层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Args:
            snr_db: [B] 物理 SNR 值 (dB, 未归一化)

        Returns:
            [B, mlp_dim]
        """
        if snr_db.dim() != 1:
            raise ValueError(f"期望 snr_db shape [B]，实际 {tuple(snr_db.shape)}")

        snr_db_f = snr_db.to(self.freqs.dtype)  # [B]
        # 归一化到 [-1, 1]
        snr_norm = (snr_db_f - self.snr_center) / self.snr_half_range  # [B]

        emb = _sinusoidal_embedding(snr_norm, self.freqs)  # [B, embed_dim]
        out = self.mlp(emb)  # [B, mlp_dim]
        return out


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.models.embeddings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8

    # ===== TimestepEmbedding =====
    print("=" * 60)
    print("TimestepEmbedding")
    print("=" * 60)
    t_emb = TimestepEmbedding(embed_dim=128, mlp_dim=64)
    n_params = sum(p.numel() for p in t_emb.parameters())
    print(f"参数量: {n_params}  (期望: 128*64 + 64 + 64*64 + 64 = {128*64 + 64 + 64*64 + 64})")

    t = torch.randint(0, 1000, (B,))  # [B]
    out = t_emb(t)
    print(f"input  t shape: {tuple(t.shape)}, dtype: {t.dtype}")
    print(f"output shape  : {tuple(out.shape)}  (期望 [{B}, 64])")
    print(f"output stats  : mean={out.mean():.4f}, std={out.std():.4f}")

    # 不同 t 应得到不同 embedding (L2 距离仅供参考，不要求单调)
    # 注: 标准 transformer 时间步 embedding 不保证 L2 距离单调，
    # 网络利用的是 embedding 向量本身在高维空间的位置而非 L2 距离
    t_pair = torch.tensor([0, 1, 500, 999])
    emb_pair = t_emb(t_pair)
    print(f"\n时间步 embedding L2 距离 (参考):")
    print(f"  t=0  vs t=1   : {(emb_pair[0] - emb_pair[1]).norm().item():.4f}  (邻近, 应较小)")
    print(f"  t=0  vs t=500 : {(emb_pair[0] - emb_pair[2]).norm().item():.4f}")
    print(f"  t=0  vs t=999 : {(emb_pair[0] - emb_pair[3]).norm().item():.4f}")
    # 关键: t=0 vs t=1 必须 << t=0 vs t=500 (邻近可区分但远离更不同)
    d01 = (emb_pair[0] - emb_pair[1]).norm().item()
    d0500 = (emb_pair[0] - emb_pair[2]).norm().item()
    print(f"  邻近 << 远距 (t=0 vs t=1 < t=0 vs t=500): {'✓' if d01 < d0500 else '✗'}")

    # ===== SNREmbedding =====
    print("\n" + "=" * 60)
    print("SNREmbedding")
    print("=" * 60)
    snr_emb = SNREmbedding(embed_dim=128, mlp_dim=64, snr_min=-15.0, snr_max=10.0)
    n_params = sum(p.numel() for p in snr_emb.parameters())
    print(f"参数量: {n_params}  (期望: {128*64 + 64 + 64*64 + 64})")

    snr = torch.tensor([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, -15.0, 10.0])  # [B]
    out = snr_emb(snr)
    print(f"input  snr shape: {tuple(snr.shape)}, dtype: {snr.dtype}")
    print(f"output shape    : {tuple(out.shape)}  (期望 [{B}, 64])")
    print(f"output stats    : mean={out.mean():.4f}, std={out.std():.4f}")

    # 测试归一化范围: -15 应映射到 -1, 10 应映射到 +1
    snr_test = torch.tensor([-15.0, 10.0])
    snr_norm = (snr_test - snr_emb.snr_center) / snr_emb.snr_half_range
    print(f"\n归一化检查: SNR=-15 -> {snr_norm[0].item():.4f} (期望 -1.0)")
    print(f"            SNR=+10 -> {snr_norm[1].item():.4f} (期望 +1.0)")

    # 平滑性检查: 在 SNR 邻域内 (差异 ≤ 5dB) 距离应严格单调递增
    # 注: sinusoidal embedding 不保证全局单调，这是其周期性本质决定的
    snr_local = torch.tensor([-15.0, -14.5, -14.0, -13.0, -12.0, -10.0])
    emb_local = snr_emb(snr_local)
    print(f"\nSNR 局部邻域单调性检查 (相对 -15.0 的 L2 距离):")
    dists = []
    for i in range(1, len(snr_local)):
        d = (emb_local[0] - emb_local[i]).norm().item()
        dists.append(d)
        delta = snr_local[i].item() - snr_local[0].item()
        print(f"  -15.0 vs {snr_local[i].item():+6.1f} (Δ={delta:+.1f}dB): {d:.4f}")
    is_local_mono = all(dists[i] < dists[i+1] for i in range(len(dists)-1))
    print(f"  局部 (Δ≤5dB) 距离单调递增: {'✓' if is_local_mono else '✗ (异常!)'}")

    # 全局参考: 远距离 SNR embedding 距离 (不要求单调，仅供参考)
    snr_global = torch.tensor([-15.0, 0.0, 10.0])
    emb_global = snr_emb(snr_global)
    print(f"\nSNR 全局参考距离 (sinusoidal 本质周期性, 不要求单调):")
    print(f"  -15.0 vs   0.0 (Δ=15dB): {(emb_global[0]-emb_global[1]).norm().item():.4f}")
    print(f"  -15.0 vs +10.0 (Δ=25dB): {(emb_global[0]-emb_global[2]).norm().item():.4f}")

    # ===== 反向传播测试 =====
    print("\n" + "=" * 60)
    print("反向传播测试")
    print("=" * 60)
    t = torch.randint(0, 1000, (B,))
    snr = torch.linspace(-15, 10, B)

    out_t = t_emb(t)
    out_s = snr_emb(snr)
    loss = (out_t.sum() + out_s.sum())
    loss.backward()

    grad_ok_t = all(p.grad is not None and p.grad.abs().sum() > 0 for p in t_emb.parameters())
    grad_ok_s = all(p.grad is not None and p.grad.abs().sum() > 0 for p in snr_emb.parameters())
    print(f"TimestepEmbedding 梯度: {'✓' if grad_ok_t else '✗'}")
    print(f"SNREmbedding      梯度: {'✓' if grad_ok_s else '✗'}")

    # ===== freqs 是 buffer，不是 parameter =====
    is_buffer_t = "freqs" in dict(t_emb.named_buffers())
    is_param_t = "freqs" in dict(t_emb.named_parameters())
    print(f"\nTimestepEmbedding.freqs 是 buffer: {is_buffer_t} (期望 True)")
    print(f"TimestepEmbedding.freqs 是 parameter: {is_param_t} (期望 False)")

    # ===== GPU 测试 =====
    if torch.cuda.is_available():
        t_emb_gpu = t_emb.cuda()
        t_gpu = t.cuda()
        out_gpu = t_emb_gpu(t_gpu)
        print(f"\n[GPU] TimestepEmbedding 输出 device: {out_gpu.device}")
