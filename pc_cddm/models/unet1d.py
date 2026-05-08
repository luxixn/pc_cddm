"""
1D-UNet for IQ signal denoising in PC-CDDM.

架构概览:
    输入 x_t [B, 2, 1024] + 含噪观测 y [B, 2, 1024] + 条件 c [B, cond_dim]
        -> 入口 Conv (4 -> base_channels): [x_t || y] concat 后送入
        -> Down 路径 (3 stages, 每 stage 2 ResBlock + 1 Downsample)
        -> Mid (2 ResBlock)
        -> Up 路径 (3 stages, 每 stage 2 ResBlock + 1 Upsample, 含 skip concat)
        -> 出口 GN + SiLU + Conv (base_channels -> 2)
    输出 ε̂ [B, 2, 1024]

关键改动 (vs. 初版):
    将含噪观测 y 作为额外输入通道与 x_t concat 后送入入口 Conv。
    初版仅靠 FiLM 条件 (SNR + PSD) 注入观测信息, 模型无法看到具体 y, 退化为
    "条件生成器" 而非去噪器: 各 SNR 档位输出 SNR 锁死同一值, NMSE > 1。
    现修复后模型每步反向去噪都能直接访问 y, 真正实现观测条件去噪。

ResBlock 采用 pre-activation 风格 (He et al., 2016 "Identity Mappings"):
    GN -> FiLM₁ -> SiLU -> Conv -> GN -> FiLM₂ -> SiLU -> [Dropout] -> Conv -> +skip
    残差路径完全无激活, 梯度直接流回, 训练稳定。
    GroupNorm 使用 affine=False, 让 FiLM 完全接管 per-channel 调制。

Skip 连接:
    Down 路径每个 ResBlock 输出 push 到栈, Up 路径每个 ResBlock 前 pop 并 concat。
    总 skip 数 = down_stages * resblocks_per_stage = 3 * 2 = 6。

参数量目标: ~10M (新增 entry conv 输入通道翻倍带来约 +128 参数, 可忽略)
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM


# ============================================================================
# ResBlock1D
# ============================================================================
class ResBlock1D(nn.Module):
    """
    Pre-activation residual block with dual FiLM injection.

    结构:
        h --> [GN -> FiLM₁ -> SiLU -> Conv]
           --> [GN -> FiLM₂ -> SiLU -> (Dropout) -> Conv]
           --> + (residual, with optional 1x1 skip projection)

    Args:
        in_channels:  C_in
        out_channels: C_out
        cond_dim:     条件向量维度 (= ConditionEncoder.cond_dim)
        gn_groups:    GroupNorm 组数
        dropout:      dropout 概率, 默认 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        gn_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # GroupNorm 要求 channels 能被 groups 整除
        if in_channels % gn_groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) 不能被 gn_groups ({gn_groups}) 整除"
            )
        if out_channels % gn_groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) 不能被 gn_groups ({gn_groups}) 整除"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一段: GN -> FiLM₁ -> SiLU -> Conv (in_channels -> out_channels)
        # GroupNorm 用 affine=False, 让 FiLM 接管 per-channel 调制
        self.norm1 = nn.GroupNorm(gn_groups, in_channels, affine=False)
        self.film1 = FiLM(cond_dim=cond_dim, channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # 第二段: GN -> FiLM₂ -> SiLU -> (Dropout) -> Conv (out_channels -> out_channels)
        self.norm2 = nn.GroupNorm(gn_groups, out_channels, affine=False)
        self.film2 = FiLM(cond_dim=cond_dim, channels=out_channels)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip projection: 升/降通道时用 1x1 Conv, 否则恒等
        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, C_in, L]
            c: [B, cond_dim]

        Returns:
            [B, C_out, L]
        """
        # 残差路径 (无激活, 无归一化)
        skip = self.skip_proj(h)  # [B, C_out, L]

        # 主路径 第一段
        x = self.norm1(h)            # [B, C_in, L]
        x = self.film1(x, c)         # [B, C_in, L]
        x = F.silu(x)                # [B, C_in, L]
        x = self.conv1(x)            # [B, C_out, L]

        # 主路径 第二段
        x = self.norm2(x)            # [B, C_out, L]
        x = self.film2(x, c)         # [B, C_out, L]
        x = F.silu(x)                # [B, C_out, L]
        x = self.dropout(x)          # [B, C_out, L]
        x = self.conv2(x)            # [B, C_out, L]

        return x + skip              # [B, C_out, L]


# ============================================================================
# Downsample / Upsample
# ============================================================================
class Downsample1D(nn.Module):
    """
    学习型下采样: Conv1d(C, C, k=4, s=2, p=1)。L -> L/2, 通道不变。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B, C, L/2]
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    上采样: nearest 插值 ×2 + Conv1d (可降通道, 比 ConvTranspose 更稳定)。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, L] -> [B, C_in, 2L] -> [B, C_out, 2L]
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


# ============================================================================
# UNet1D
# ============================================================================
class UNet1D(nn.Module):
    """
    1D-UNet 主体, 接收 x_t、含噪观测 y 和条件 c, 预测噪声 ε̂。

    架构 (默认参数):
        in_ch=2, base=64, mults=[1,2,4], num_res=2, gn_groups=8

        Entry:  Conv(4 -> 64)  ← [x_t (2ch) || y (2ch)] concat 后送入
        Down 0: ResBlock(64->64), ResBlock(64->64), Downsample(64)        L: 1024->512
        Down 1: ResBlock(64->128), ResBlock(128->128), Downsample(128)    L: 512->256
        Down 2: ResBlock(128->256), ResBlock(256->256)  (无 downsample)
        Mid:    ResBlock(256->256), ResBlock(256->256)
        Up 2:   ResBlock(256+256->256), ResBlock(256+256->256), Upsample(256->128)  L: 256->512
        Up 1:   ResBlock(128+128->128), ResBlock(128+128->128), Upsample(128->64)   L: 512->1024
        Up 0:   ResBlock(64+64->64), ResBlock(64+64->64)
        Exit:   GN(64) -> SiLU -> Conv(64 -> 2)

    Args:
        in_channels:    单路输入通道数 (= 2, IQ 双通道; 实际 entry 输入 2*in_channels=4)
        base_channels:  基础通道数 (= 64)
        channel_mults:  各级通道倍率 (= [1, 2, 4])
        num_res_blocks: 每级 ResBlock 数 (= 2)
        cond_dim:       条件向量维度 (= 256)
        gn_groups:      GroupNorm 组数 (= 8)
        dropout:        ResBlock 内 dropout (默认 0)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        gn_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.cond_dim = cond_dim

        n_stages = len(channel_mults)

        # 各 stage 的通道数 [64, 128, 256] (默认)
        stage_channels = [base_channels * m for m in channel_mults]

        # ----- Entry -----
        # 输入是 [x_t || y] 在 channel 维 concat, 通道数 = 2 * in_channels
        self.entry = nn.Conv1d(in_channels * 2, base_channels, kernel_size=3, padding=1)

        # ----- Down 路径 -----
        # 注意: 第一个 stage 输入是 base_channels (来自 entry), 不是 stage_channels[0]
        # 但默认 channel_mults[0]=1, 所以 stage_channels[0] = base, 一致
        self.down_blocks = nn.ModuleList()       # 各 stage 的 ResBlock 列表
        self.down_samples = nn.ModuleList()      # 各 stage 的 Downsample (最后一个 stage 是 Identity)

        ch_in = base_channels
        for i, ch_out in enumerate(stage_channels):
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # 第一个 ResBlock 升通道 (ch_in -> ch_out), 后续保持
                in_c = ch_in if j == 0 else ch_out
                stage_blocks.append(
                    ResBlock1D(in_c, ch_out, cond_dim, gn_groups, dropout)
                )
            self.down_blocks.append(stage_blocks)

            # 最后一个 stage 不再下采样
            if i < n_stages - 1:
                self.down_samples.append(Downsample1D(ch_out))
            else:
                self.down_samples.append(nn.Identity())

            ch_in = ch_out

        # ----- Mid -----
        mid_ch = stage_channels[-1]  # 256
        self.mid_blocks = nn.ModuleList([
            ResBlock1D(mid_ch, mid_ch, cond_dim, gn_groups, dropout),
            ResBlock1D(mid_ch, mid_ch, cond_dim, gn_groups, dropout),
        ])

        # ----- Up 路径 (镜像 Down) -----
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # Up 从最深 stage 开始, 反向遍历
        ch_in = stage_channels[-1]  # mid 出来 = 256
        for i in reversed(range(n_stages)):
            ch_out = stage_channels[i]              # 当前 stage 的输出通道
            ch_skip = stage_channels[i]             # 来自 down 的 skip 通道 (与 stage 一致)

            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # 每个 ResBlock 输入 = 上一层 + skip (concat)
                # 第一个 ResBlock: ch_in (来自 mid 或 上一 upsample) + ch_skip
                # 后续 ResBlock: ch_out (上一 ResBlock 输出) + ch_skip
                cat_in = (ch_in if j == 0 else ch_out) + ch_skip
                stage_blocks.append(
                    ResBlock1D(cat_in, ch_out, cond_dim, gn_groups, dropout)
                )
            self.up_blocks.append(stage_blocks)

            # 最后一个 up stage (i=0) 不再上采样
            if i > 0:
                # 上采样到下一级 stage_channels[i-1]
                self.up_samples.append(Upsample1D(ch_out, stage_channels[i - 1]))
                ch_in = stage_channels[i - 1]
            else:
                self.up_samples.append(nn.Identity())

        # ----- Exit -----
        self.exit_norm = nn.GroupNorm(gn_groups, base_channels, affine=True)
        # exit 这里用 affine=True, 因为已脱离条件控制路径, 让 GN 自学一组 affine
        self.exit_conv = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

        # 出口 Conv 的权重清零, 让网络初始预测 ε̂ ≈ 0
        # 这是 DDPM 标准做法, 保证训练初期 loss 在合理范围
        nn.init.zeros_(self.exit_conv.weight)
        nn.init.zeros_(self.exit_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, L] noisy IQ signal x_t (扩散链中间态)
            c: [B, cond_dim]       condition vector (SNR + PSD + t)
            y: [B, in_channels, L] 含噪观测 IQ 信号 (反向链中保持不变)

        Returns:
            [B, in_channels, L]    predicted noise ε̂
        """
        if x.dim() != 3 or x.size(1) != self.in_channels:
            raise ValueError(
                f"期望 x shape [B, {self.in_channels}, L]，实际 {tuple(x.shape)}"
            )
        if y.shape != x.shape:
            raise ValueError(
                f"y 必须与 x 同 shape, 期望 {tuple(x.shape)}, 实际 {tuple(y.shape)}"
            )
        if c.dim() != 2 or c.size(1) != self.cond_dim:
            raise ValueError(
                f"期望 c shape [B, {self.cond_dim}]，实际 {tuple(c.shape)}"
            )

        # ----- Entry: 拼接 x_t 与 y 后送入 -----
        h_in = torch.cat([x, y], dim=1)  # [B, 2*in_channels, L]
        h = self.entry(h_in)              # [B, base_channels, L]

        # ----- Down -----
        skips: list[torch.Tensor] = []
        for stage_blocks, down in zip(self.down_blocks, self.down_samples):
            for resblock in stage_blocks:
                h = resblock(h, c)
                skips.append(h)  # 存 ResBlock 输出 (downsample 之前)
            h = down(h)          # 下采样 (最后 stage 是 Identity)

        # ----- Mid -----
        for resblock in self.mid_blocks:
            h = resblock(h, c)

        # ----- Up -----
        for stage_blocks, up in zip(self.up_blocks, self.up_samples):
            for resblock in stage_blocks:
                skip = skips.pop()  # LIFO, 与 down 顺序匹配
                h = torch.cat([h, skip], dim=1)  # cat 在 channel 维
                h = resblock(h, c)
            h = up(h)  # 上采样 (最后 stage 是 Identity)

        # 校验所有 skip 都用完了
        assert len(skips) == 0, f"skip 未对齐, 剩余 {len(skips)}"

        # ----- Exit -----
        h = self.exit_norm(h)        # [B, base_channels, L]
        h = F.silu(h)
        out = self.exit_conv(h)      # [B, in_channels, L]
        return out

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> "UNet1D":
        """
        从 yaml 配置字典构造。

        Args:
            model_cfg: yaml['model'] 段
        """
        return cls(
            in_channels=2,  # 固定 IQ 双通道 (entry conv 内部会乘 2 接收 x_t||y)
            base_channels=model_cfg["base_channels"],
            channel_mults=tuple(model_cfg["channel_mults"]),
            num_res_blocks=model_cfg["num_res_blocks"],
            cond_dim=model_cfg["cond_dim"],
            gn_groups=model_cfg["groupnorm_groups"],
            dropout=model_cfg.get("dropout", 0.0),
        )


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.models.unet1d
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # ===== 默认配置构造 =====
    print("=" * 60)
    print("UNet1D 默认配置 (base=64, mults=[1,2,4], num_res=2)")
    print("=" * 60)
    net = UNet1D(
        in_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        cond_dim=256,
        gn_groups=8,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {n_params:,}  (相比初版 6,571,458 多约 +128 个 entry conv 参数)")

    # 各模块参数量分解
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    print(f"  entry           : {count_params(net.entry):>10,}")
    print(f"  down_blocks     : {count_params(net.down_blocks):>10,}")
    print(f"  down_samples    : {count_params(net.down_samples):>10,}")
    print(f"  mid_blocks      : {count_params(net.mid_blocks):>10,}")
    print(f"  up_blocks       : {count_params(net.up_blocks):>10,}")
    print(f"  up_samples      : {count_params(net.up_samples):>10,}")
    print(f"  exit            : {count_params(net.exit_norm) + count_params(net.exit_conv):>10,}")

    # ===== Forward 测试 =====
    B, L = 4, 1024
    x = torch.randn(B, 2, L)
    c = torch.randn(B, 256)
    y = torch.randn(B, 2, L)

    out = net(x, c, y)
    print(f"\n输入 x shape: {tuple(x.shape)}")
    print(f"输入 y shape: {tuple(y.shape)}")
    print(f"输入 c shape: {tuple(c.shape)}")
    print(f"输出 shape  : {tuple(out.shape)}  (期望 [{B}, 2, {L}])")
    print(f"输出 stats  : mean={out.mean():.6f}, std={out.std():.6f}")
    print(f"  注: 出口 Conv 零初始化, 训练前输出应为 0")
    is_zero_init = out.abs().max().item() < 1e-6
    print(f"  零初始化检查: {'✓' if is_zero_init else '✗'}")

    # ===== 训练后输出非零 =====
    with torch.no_grad():
        net.exit_conv.weight.normal_(0, 0.01)
    out2 = net(x, c, y)
    print(f"\n模拟训练后输出 std: {out2.std():.6f} (应 > 0)")
    print(f"  非零检查: {'✓' if out2.std() > 1e-4 else '✗'}")

    # ===== 反向传播 =====
    print(f"\n反向传播测试:")
    net.zero_grad()
    out3 = net(x, c, y)
    loss = out3.pow(2).mean()
    loss.backward()
    no_grad = [n for n, p in net.named_parameters() if p.grad is None]
    print(f"  无梯度参数: {len(no_grad)} 个 (期望 0)")
    if no_grad:
        for n in no_grad[:5]:
            print(f"    {n}")

    # ===== 不同输入产生不同输出 (在模拟训练状态下测, 因为初始 zero-init) =====
    # 给 exit_conv 和部分 FiLM 赋随机权重, 模拟训练后状态
    with torch.no_grad():
        net.exit_conv.weight.normal_(0, 0.01)
        net.exit_conv.bias.normal_(0, 0.01)
        # 给所有 FiLM 的 to_gamma/to_beta 一些小权重, 让 c 真正影响输出
        for m in net.modules():
            if isinstance(m, FiLM):
                m.to_gamma.weight.normal_(0, 0.05)
                m.to_beta.weight.normal_(0, 0.05)

    net.zero_grad()
    x1 = torch.randn(B, 2, L)
    x2 = torch.randn(B, 2, L)
    y_same = torch.randn(B, 2, L)
    c_same = torch.randn(B, 256)
    out_x1 = net(x1, c_same, y_same)
    out_x2 = net(x2, c_same, y_same)
    diff_x = (out_x1 - out_x2).abs().mean().item()
    print(f"\n[模拟训练后] x 敏感性: 不同 x 同 c 同 y, mean|Δout| = {diff_x:.4f}")
    print(f"  期望 > 0: {'✓' if diff_x > 1e-4 else '✗'}")

    # 不同观测 y 产生不同输出 (新增检查, 验证 y 真正进入网络)
    x_same = torch.randn(B, 2, L)
    y1 = torch.randn(B, 2, L)
    y2 = torch.randn(B, 2, L)
    out_y1 = net(x_same, c_same, y1)
    out_y2 = net(x_same, c_same, y2)
    diff_y = (out_y1 - out_y2).abs().mean().item()
    print(f"[模拟训练后] y 敏感性: 同 x 同 c 不同 y, mean|Δout| = {diff_y:.4f}")
    print(f"  期望 > 0 (y 必须真正影响输出): {'✓' if diff_y > 1e-4 else '✗'}")

    # 不同条件产生不同输出
    c1 = torch.randn(B, 256)
    c2 = torch.randn(B, 256)
    out_c1 = net(x_same, c1, y_same)
    out_c2 = net(x_same, c2, y_same)
    diff_c = (out_c1 - out_c2).abs().mean().item()
    print(f"[模拟训练后] c 敏感性: 同 x 同 y 不同 c, mean|Δout| = {diff_c:.4f}")
    print(f"  期望 > 0: {'✓' if diff_c > 1e-4 else '✗'}")
    print(f"  注: 初始 zero-init 时条件无影响是设计预期, 训练后 FiLM 学到非零调制即可")

    # ===== Shape 链路追踪 =====
    print(f"\nShape 链路追踪 (B=2, L=1024):")
    net.eval()
    x_trace = torch.randn(2, 2, 1024)
    c_trace = torch.randn(2, 256)
    y_trace = torch.randn(2, 2, 1024)
    with torch.no_grad():
        h_in = torch.cat([x_trace, y_trace], dim=1)
        print(f"  entry input (cat): {tuple(h_in.shape)}")
        h = net.entry(h_in)
        print(f"  entry out      : {tuple(h.shape)}")
        for i, (stage, down) in enumerate(zip(net.down_blocks, net.down_samples)):
            for j, rb in enumerate(stage):
                h = rb(h, c_trace)
                print(f"  down[{i}].res[{j}]  : {tuple(h.shape)}")
            h = down(h)
            if not isinstance(down, nn.Identity):
                print(f"  down[{i}].sample : {tuple(h.shape)}")

    # ===== from_config 测试 =====
    print(f"\nfrom_config 测试:")
    fake_cfg = {
        "base_channels": 64,
        "channel_mults": [1, 2, 4],
        "num_res_blocks": 2,
        "cond_dim": 256,
        "groupnorm_groups": 8,
        "dropout": 0.0,
    }
    net2 = UNet1D.from_config(fake_cfg)
    n2 = sum(p.numel() for p in net2.parameters())
    print(f"  构造成功, 参数量: {n2:,} (应等于 {n_params:,})")

    # ===== 显存友好性测试 (不实际跑, 估算) =====
    # forward + backward 显存粗估: 2 * (中间激活总量 * 4B)
    # base=64, L=1024, 三级 [1024, 512, 256] -> 通道 [64, 128, 256]
    # 每级激活量 ~ B * C * L
    print(f"\n显存粗估 (B=32, FP32):")
    act_64_1024 = 32 * 64 * 1024 * 4 / 1e6   # MB
    act_128_512 = 32 * 128 * 512 * 4 / 1e6
    act_256_256 = 32 * 256 * 256 * 4 / 1e6
    total_act = (act_64_1024 + act_128_512 + act_256_256) * 6  # 粗略乘 6 (forward + backward + skip)
    print(f"  单级激活量: {act_64_1024:.1f}/{act_128_512:.1f}/{act_256_256:.1f} MB")
    print(f"  forward+backward 估计: ~{total_act:.0f} MB (粗略)")

    # ===== GPU 测试 =====
    if torch.cuda.is_available():
        net_gpu = UNet1D(
            in_channels=2, base_channels=64, channel_mults=(1, 2, 4),
            num_res_blocks=2, cond_dim=256, gn_groups=8, dropout=0.0,
        ).cuda()
        x_gpu = torch.randn(B, 2, L).cuda()
        c_gpu = torch.randn(B, 256).cuda()
        y_gpu = torch.randn(B, 2, L).cuda()
        out_gpu = net_gpu(x_gpu, c_gpu, y_gpu)
        print(f"\n[GPU] 输出 device: {out_gpu.device}, shape: {tuple(out_gpu.shape)}")
