"""
FiLM (Feature-wise Linear Modulation) module.

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.

核心机制:
    给定条件向量 c ∈ R^cond_dim 和 feature map h ∈ R^{B×C×L}，
    生成两组逐通道仿射参数 γ ∈ R^{B×C}, β ∈ R^{B×C}，
    输出: h' = γ(c) · h + β(c)
    其中 γ 和 β 沿通道维 (dim=1) 作用于 h，沿空间维 (dim=2) 广播。

设计要点:
    1. γ 和 β 使用独立 Linear 层（更灵活，参数量可控）。
    2. 初始化为恒等变换 h' = 1·h + 0，避免训练初期破坏 backbone 信号:
        - γ 输出 = 1 + Linear(c)，Linear 默认初始化使输出接近 0，γ 起点 ≈ 1
        - β 输出 = Linear(c)，默认接近 0
    3. 在 ResBlock 中通常调用两次（GroupNorm 之后、SiLU 之前），
       每次实例化独立 FiLM 对象（参数不共享）。
    4. 张量布局假设: feature map 为 [B, C, L]（Conv1d 标准）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    Args:
        cond_dim: 条件向量维度（通常 = condition_encoder 输出的 cond_dim）
        channels: feature map 通道数 C

    Forward:
        h: [B, C, L]      feature map
        c: [B, cond_dim]  condition vector
    Returns:
        h': [B, C, L]     modulated feature map
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.channels = channels

        # γ 和 β 各自一个 Linear: cond_dim -> channels
        # 参数量: 2 * (cond_dim * channels + channels) ≈ 2·cond_dim·C
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)

        # 恒等初始化: γ 起点 ≈ 1, β 起点 ≈ 0
        # PyTorch Linear 默认 kaiming uniform 初始化，权重小但非零，输出接近 0。
        # 显式将 to_gamma 和 to_beta 的权重清零、bias 清零，使初始输出严格为 0,
        # 然后在 forward 里 γ = 1 + to_gamma(c) 实现 γ_init = 1, β_init = 0。
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, C, L] feature map
            c: [B, cond_dim] condition vector

        Returns:
            [B, C, L] modulated feature map
        """
        if h.dim() != 3:
            raise ValueError(f"期望 h shape [B, C, L]，实际 {tuple(h.shape)}")
        if c.dim() != 2:
            raise ValueError(f"期望 c shape [B, cond_dim]，实际 {tuple(c.shape)}")
        if h.size(0) != c.size(0):
            raise ValueError(
                f"batch 维不匹配: h.B={h.size(0)}, c.B={c.size(0)}"
            )
        if h.size(1) != self.channels:
            raise ValueError(
                f"通道数不匹配: h.C={h.size(1)}, expected C={self.channels}"
            )
        if c.size(1) != self.cond_dim:
            raise ValueError(
                f"条件维度不匹配: c.dim={c.size(1)}, expected cond_dim={self.cond_dim}"
            )

        # 生成 γ, β: [B, cond_dim] -> [B, C]
        gamma = 1.0 + self.to_gamma(c)  # [B, C]  恒等初始化: γ_init = 1
        beta = self.to_beta(c)          # [B, C]  恒等初始化: β_init = 0

        # 增加空间维度以便广播: [B, C] -> [B, C, 1]
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)    # [B, C, 1]

        # 广播相乘相加: [B, C, L] * [B, C, 1] + [B, C, 1] -> [B, C, L]
        return gamma * h + beta


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.models.film
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, L = 4, 64, 128
    cond_dim = 256

    film = FiLM(cond_dim=cond_dim, channels=C)
    n_params = sum(p.numel() for p in film.parameters())
    expected = 2 * (cond_dim * C + C)
    print(f"参数量: {n_params}  (期望 {expected})")

    h = torch.randn(B, C, L)
    c = torch.randn(B, cond_dim)

    h_out = film(h, c)

    print(f"\n输入 h shape : {tuple(h.shape)}")
    print(f"输入 c shape : {tuple(c.shape)}")
    print(f"输出   shape : {tuple(h_out.shape)}  (期望 [{B}, {C}, {L}])")

    # ===== 关键: 恒等初始化测试 =====
    # 训练初期 (权重/bias 全为 0), γ=1, β=0, 应该有 h_out == h
    diff = (h_out - h).abs().max().item()
    print(f"\n恒等初始化测试: max|h_out - h| = {diff:.6e}")
    print(f"  期望 < 1e-6 (严格恒等), 实际: {'✓' if diff < 1e-6 else '✗'}")

    # ===== 训练后 FiLM 应该改变特征 =====
    # 模拟训练: 给 to_gamma/to_beta 赋随机权重
    with torch.no_grad():
        film.to_gamma.weight.normal_(0, 0.1)
        film.to_gamma.bias.normal_(0, 0.1)
        film.to_beta.weight.normal_(0, 0.1)
        film.to_beta.bias.normal_(0, 0.1)
    h_out2 = film(h, c)
    diff2 = (h_out2 - h).abs().mean().item()
    print(f"\n模拟训练后: mean|h_out - h| = {diff2:.4f}")
    print(f"  期望 > 0.01 (FiLM 起作用), 实际: {'✓' if diff2 > 0.01 else '✗'}")

    # ===== 广播正确性测试 =====
    # 同一个 c, 不同空间位置 L 处的 γ/β 应该完全相同 (per-channel, 不per-spatial)
    # 即: h_out[:, :, l1] - h_out[:, :, l2] 应该等于 γ * (h[:, :, l1] - h[:, :, l2])
    # 验证方式: 把 h 在 L 维上常数化, 检查 h_out 在 L 维上也保持常数
    h_const = torch.randn(B, C, 1).expand(B, C, L).contiguous()  # [B, C, L] 但 L 维都相同
    h_out3 = film(h_const, c)
    L_var = h_out3.var(dim=-1).max().item()  # L 维的方差应该是 0
    print(f"\n广播正确性: 输入 L 维常数时, 输出 L 维方差 = {L_var:.6e}")
    print(f"  期望 < 1e-6 (γ/β 沿 L 维广播), 实际: {'✓' if L_var < 1e-6 else '✗'}")

    # ===== 反向传播测试 =====
    film2 = FiLM(cond_dim=cond_dim, channels=C)
    h = torch.randn(B, C, L, requires_grad=True)
    c = torch.randn(B, cond_dim, requires_grad=True)
    h_out = film2(h, c)
    loss = h_out.sum()
    loss.backward()

    grad_h_ok = h.grad is not None
    grad_c_ok = c.grad is not None
    grad_film_ok = all(p.grad is not None for p in film2.parameters())
    # 注意: 此时 to_gamma/to_beta 仍是零初始化, c 的梯度可能为 0
    # (因为 ∂loss/∂c 经过零权重的 Linear 后变 0)
    # 这是预期的——零初始化下条件输入"暂时"不影响输出
    print(f"\n反向传播测试:")
    print(f"  h.grad     : {'✓' if grad_h_ok else '✗'}")
    print(f"  c.grad     : {'✓' if grad_c_ok else '✗'} (零初始化下值可能为0, 是正常的)")
    print(f"  film.grad  : {'✓' if grad_film_ok else '✗'}")
    # 梯度应能流到 to_gamma/to_beta 的权重 (它们的 grad 不应是 None)
    print(f"  to_gamma.weight.grad norm: {film2.to_gamma.weight.grad.norm().item():.4f}")
    print(f"  to_beta.weight.grad norm : {film2.to_beta.weight.grad.norm().item():.4f}")

    # ===== 不同 batch 不同条件应得到不同输出 =====
    film3 = FiLM(cond_dim=cond_dim, channels=C)
    with torch.no_grad():
        film3.to_gamma.weight.normal_(0, 0.1)
        film3.to_beta.weight.normal_(0, 0.1)
    h_same = torch.randn(1, C, L).expand(B, C, L).contiguous()  # batch 内 h 相同
    c_diff = torch.randn(B, cond_dim)                           # batch 内 c 不同
    h_out4 = film3(h_same, c_diff)
    batch_var = h_out4.var(dim=0).mean().item()  # batch 维的方差应该 > 0
    print(f"\nBatch 独立性: 不同 c 下输出 batch 维方差 = {batch_var:.4f}")
    print(f"  期望 > 0 (不同条件产生不同输出), 实际: {'✓' if batch_var > 1e-4 else '✗'}")

    # ===== GPU 测试 =====
    if torch.cuda.is_available():
        film_gpu = FiLM(cond_dim=cond_dim, channels=C).cuda()
        h_gpu = torch.randn(B, C, L).cuda()
        c_gpu = torch.randn(B, cond_dim).cuda()
        h_out_gpu = film_gpu(h_gpu, c_gpu)
        print(f"\n[GPU] 输出 device: {h_out_gpu.device}, shape: {tuple(h_out_gpu.shape)}")
