"""
Condition Encoder: 融合时间步、SNR、PSD 三路条件信息为最终条件向量 c。

输入:
    t:      [B]              扩散时间步 (long int, 0..T-1)
    snr_db: [B]              物理 SNR (float, dB)
    psd:    [B, psd_input_dim] log 域双边 PSD (来自 utils.psd.welch_psd_log)

输出:
    c:      [B, cond_dim]    条件向量, 供 FiLM 注入 1D-UNet 各 ResBlock

架构:
    t      --[TimestepEmbedding]-->        [B, time_mlp_dim]   ┐
    snr_db --[SNREmbedding]-->             [B, snr_mlp_dim]    ├-> concat -> [B, sum]
    psd    --[Linear+SiLU+Linear]-->       [B, psd_mlp_out]    ┘
                                                                  │
                                                            fusion MLP (Linear+SiLU+Linear)
                                                                  ▼
                                                              [B, cond_dim] = c
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from .embeddings import TimestepEmbedding, SNREmbedding


class ConditionEncoder(nn.Module):
    """
    三路条件融合编码器。

    Args:
        # 时间步分支
        time_embed_dim:  TimestepEmbedding 的 sinusoidal 展开维度
        time_mlp_dim:    TimestepEmbedding 输出 MLP 维度
        # SNR 分支
        snr_embed_dim:   SNREmbedding 的 sinusoidal 展开维度
        snr_mlp_dim:     SNREmbedding 输出 MLP 维度
        snr_min:         SNR 归一化下界
        snr_max:         SNR 归一化上界
        # PSD 分支
        psd_input_dim:   PSD 输入维度 (= nperseg, 双边谱)
        psd_mlp_hidden:  PSD MLP 中间隐藏维度
        psd_mlp_out:     PSD MLP 输出维度
        # 融合
        cond_dim:        最终条件向量维度
    """

    def __init__(
        self,
        # 时间步
        time_embed_dim: int = 128,
        time_mlp_dim: int = 64,
        # SNR
        snr_embed_dim: int = 128,
        snr_mlp_dim: int = 64,
        snr_min: float = -15.0,
        snr_max: float = 10.0,
        # PSD
        psd_input_dim: int = 256,
        psd_mlp_hidden: int = 256,
        psd_mlp_out: int = 64,
        # 融合
        cond_dim: int = 256,
    ):
        super().__init__()

        self.psd_input_dim = psd_input_dim
        self.cond_dim = cond_dim

        # ===== 三路分支 =====
        # 时间步
        self.t_embed = TimestepEmbedding(
            embed_dim=time_embed_dim,
            mlp_dim=time_mlp_dim,
        )
        # SNR
        self.snr_embed = SNREmbedding(
            embed_dim=snr_embed_dim,
            mlp_dim=snr_mlp_dim,
            snr_min=snr_min,
            snr_max=snr_max,
        )
        # PSD: 2 层 MLP, Linear -> SiLU -> Linear
        self.psd_mlp = nn.Sequential(
            nn.Linear(psd_input_dim, psd_mlp_hidden),
            nn.SiLU(),
            nn.Linear(psd_mlp_hidden, psd_mlp_out),
        )

        # ===== 融合 MLP =====
        concat_dim = time_mlp_dim + snr_mlp_dim + psd_mlp_out
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        snr_db: torch.Tensor,
        psd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t:      [B]                整数时间步
            snr_db: [B]                物理 SNR (dB)
            psd:    [B, psd_input_dim] log 域双边 PSD

        Returns:
            c: [B, cond_dim] 融合后的条件向量
        """
        # 输入校验
        if t.dim() != 1:
            raise ValueError(f"期望 t shape [B]，实际 {tuple(t.shape)}")
        if snr_db.dim() != 1:
            raise ValueError(f"期望 snr_db shape [B]，实际 {tuple(snr_db.shape)}")
        if psd.dim() != 2 or psd.size(1) != self.psd_input_dim:
            raise ValueError(
                f"期望 psd shape [B, {self.psd_input_dim}]，实际 {tuple(psd.shape)}"
            )
        B = t.size(0)
        if snr_db.size(0) != B or psd.size(0) != B:
            raise ValueError(
                f"batch 维不匹配: t.B={B}, snr_db.B={snr_db.size(0)}, psd.B={psd.size(0)}"
            )

        # 三路独立编码
        t_feat = self.t_embed(t)         # [B, time_mlp_dim]
        snr_feat = self.snr_embed(snr_db)  # [B, snr_mlp_dim]
        psd_feat = self.psd_mlp(psd)      # [B, psd_mlp_out]

        # 拼接 + 融合
        cat = torch.cat([t_feat, snr_feat, psd_feat], dim=-1)  # [B, sum_dim]
        c = self.fusion(cat)  # [B, cond_dim]
        return c

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any], psd_cfg: dict[str, Any]) -> "ConditionEncoder":
        """
        从 yaml 配置字典构造。

        用法:
            cfg = yaml.safe_load(open('configs/default.yaml'))
            ce = ConditionEncoder.from_config(cfg['model'], cfg['psd'])

        Args:
            model_cfg: yaml['model'] 段
            psd_cfg:   yaml['psd'] 段 (用于读取 nperseg 推断 psd_input_dim)
        """
        return cls(
            time_embed_dim=model_cfg["time_embed_dim"],
            time_mlp_dim=model_cfg["time_mlp_dim"],
            snr_embed_dim=model_cfg["snr_embed_dim"],
            snr_mlp_dim=model_cfg["snr_mlp_dim"],
            snr_min=model_cfg.get("snr_min", -15.0),
            snr_max=model_cfg.get("snr_max", 10.0),
            psd_input_dim=psd_cfg["nperseg"],  # 双边谱维度 = nperseg
            psd_mlp_hidden=model_cfg["psd_mlp_hidden"],
            psd_mlp_out=model_cfg["psd_mlp_out"],
            cond_dim=model_cfg["cond_dim"],
        )


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.models.condition_encoder
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 8
    psd_dim = 256
    cond_dim = 256

    ce = ConditionEncoder(
        time_embed_dim=128, time_mlp_dim=64,
        snr_embed_dim=128, snr_mlp_dim=64,
        snr_min=-15.0, snr_max=10.0,
        psd_input_dim=psd_dim, psd_mlp_hidden=256, psd_mlp_out=64,
        cond_dim=cond_dim,
    )
    n_params = sum(p.numel() for p in ce.parameters())
    print(f"参数量: {n_params}")

    # 输入
    t = torch.randint(0, 1000, (B,))
    snr_db = torch.linspace(-15, 10, B)  # [-15, -11.4, ..., 10]
    psd = torch.randn(B, psd_dim)        # 模拟 log PSD

    c = ce(t, snr_db, psd)
    print(f"\n输入:")
    print(f"  t      shape: {tuple(t.shape)}")
    print(f"  snr_db shape: {tuple(snr_db.shape)}")
    print(f"  psd    shape: {tuple(psd.shape)}")
    print(f"输出 c shape  : {tuple(c.shape)}  (期望 [{B}, {cond_dim}])")
    print(f"输出 c stats  : mean={c.mean():.4f}, std={c.std():.4f}")

    # ===== 三路独立性测试 =====
    # 固定其中两路, 改变第三路, 输出应该不同
    t_fixed = torch.zeros(B, dtype=torch.long)
    snr_fixed = torch.zeros(B)
    psd_fixed = torch.zeros(B, psd_dim)

    c_base = ce(t_fixed, snr_fixed, psd_fixed)

    # 只改 t
    c_dt = ce(torch.full((B,), 500, dtype=torch.long), snr_fixed, psd_fixed)
    diff_t = (c_dt - c_base).abs().mean().item()
    print(f"\n三路独立性 (改一路, 固定另两路, mean|Δc|):")
    print(f"  改 t (0->500)         : {diff_t:.4f}  {'✓' if diff_t > 1e-4 else '✗'}")

    # 只改 snr
    c_ds = ce(t_fixed, torch.full((B,), 5.0), psd_fixed)
    diff_s = (c_ds - c_base).abs().mean().item()
    print(f"  改 snr (0->5)         : {diff_s:.4f}  {'✓' if diff_s > 1e-4 else '✗'}")

    # 只改 psd
    c_dp = ce(t_fixed, snr_fixed, torch.randn(B, psd_dim))
    diff_p = (c_dp - c_base).abs().mean().item()
    print(f"  改 psd (0->random)    : {diff_p:.4f}  {'✓' if diff_p > 1e-4 else '✗'}")

    # ===== 反向传播测试 =====
    t = torch.randint(0, 1000, (B,))
    snr_db = torch.randn(B) * 5  # SNR 在 ±10 范围
    psd = torch.randn(B, psd_dim)
    c = ce(t, snr_db, psd)
    loss = c.sum()
    loss.backward()

    # 检查所有参数都有非零梯度
    no_grad_params = [n for n, p in ce.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    print(f"\n反向传播测试:")
    print(f"  无梯度参数: {len(no_grad_params)} 个 (期望 0)")
    if no_grad_params:
        print(f"  失败的参数: {no_grad_params}")
    else:
        print(f"  ✓ 所有参数梯度正常")

    # ===== 输入维度校验测试 =====
    print(f"\n输入维度校验:")
    try:
        ce(t, snr_db, torch.randn(B, 128))  # 错误的 psd 维度
        print(f"  ✗ 应该抛错但没抛")
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {str(e)[:60]}...")

    # ===== from_config 测试 =====
    print(f"\nfrom_config 测试:")
    fake_model_cfg = {
        "time_embed_dim": 128, "time_mlp_dim": 64,
        "snr_embed_dim": 128, "snr_mlp_dim": 64,
        "snr_min": -15.0, "snr_max": 10.0,
        "psd_mlp_hidden": 256, "psd_mlp_out": 64,
        "cond_dim": 256,
    }
    fake_psd_cfg = {"nperseg": 256}
    ce2 = ConditionEncoder.from_config(fake_model_cfg, fake_psd_cfg)
    n_params2 = sum(p.numel() for p in ce2.parameters())
    print(f"  from_config 构造成功, 参数量: {n_params2} (应等于 {n_params})")

    # ===== GPU 测试 =====
    if torch.cuda.is_available():
        ce_gpu = ConditionEncoder(
            psd_input_dim=psd_dim, cond_dim=cond_dim,
        ).cuda()
        t_gpu = torch.randint(0, 1000, (B,)).cuda()
        snr_gpu = torch.randn(B).cuda()
        psd_gpu = torch.randn(B, psd_dim).cuda()
        c_gpu = ce_gpu(t_gpu, snr_gpu, psd_gpu)
        print(f"\n[GPU] 输出 device: {c_gpu.device}, shape: {tuple(c_gpu.shape)}")
