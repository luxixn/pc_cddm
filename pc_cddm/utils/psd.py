"""
PSD estimation via Welch's method, GPU-native PyTorch implementation.

设计要点：
1. 输入为 [B, 2, L] 的 I/Q 双通道实信号，内部合成复信号 z = I + jQ。
2. 使用 torch.fft.rfft 在 GPU 上执行，避免 CPU 拷贝。
3. 单边谱输出 [B, nperseg//2 + 1]，默认 nperseg=256 -> 129 维。
4. log 域接口为全局主入口；原始 PSD 接口保留以备消融。

数学约定（与 scipy.signal.welch(..., return_onesided=True, scaling='density') 一致）:
    - 加 Hann 窗 w[n]
    - 每段 segment 做 FFT 后取 |X|^2
    - 归一化因子: 1 / (fs * sum(w^2))
    - 单边谱: 除 DC 和 Nyquist 外，其余频点 *2
    - 多段平均得到最终 PSD

注意: 这里采用归一化频率, fs=1.0。如需物理频率可在调用方乘以 1/fs。
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 窗函数缓存：避免每次调用都重新构造 Hann 窗
# 以 (nperseg, device, dtype) 为 key 缓存
# ---------------------------------------------------------------------------
_WINDOW_CACHE: dict[tuple, torch.Tensor] = {}


def _get_hann_window(nperseg: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    返回长度为 nperseg 的 Hann 窗，缓存复用。

    Returns:
        win: [nperseg]
    """
    key = (nperseg, device, dtype)
    if key not in _WINDOW_CACHE:
        # torch.hann_window(periodic=True) 与 scipy 'hann' 默认一致
        _WINDOW_CACHE[key] = torch.hann_window(
            nperseg, periodic=True, device=device, dtype=dtype
        )
    return _WINDOW_CACHE[key]


def _frame_signal(z: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    """
    将信号切成重叠 segment。

    Args:
        z: [B, L] 复或实信号
        nperseg: 每段长度
        noverlap: 相邻段重叠样本数

    Returns:
        frames: [B, n_segments, nperseg]
    """
    B, L = z.shape  # [B, L]
    step = nperseg - noverlap
    if L < nperseg:
        raise ValueError(
            f"信号长度 L={L} 小于 nperseg={nperseg}，无法分段。"
        )
    # unfold 在最后一维上滑窗: [B, n_segments, nperseg]
    frames = z.unfold(dimension=-1, size=nperseg, step=step)
    return frames


def welch_psd(
    x: torch.Tensor,
    nperseg: int = 256,
    noverlap: int = 128,
    fs: float = 1.0,
) -> torch.Tensor:
    """
    Welch 法估计单边 PSD（线性域）。

    输入 IQ 双通道实信号 [B, 2, L]，内部合成复信号 z = I + jQ，
    输出复信号的单边功率谱密度 [B, nperseg//2 + 1]。

    Args:
        x: [B, 2, L] I/Q 双通道实信号 (channel 0 = I, channel 1 = Q)
        nperseg: 每段长度，默认 256
        noverlap: 段间重叠样本数，默认 128
        fs: 采样率，归一化频率下取 1.0

    Returns:
        psd: [B, nperseg // 2 + 1] 单边 PSD（线性域，非负）
    """
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"期望输入 shape [B, 2, L]，实际 {tuple(x.shape)}")

    B, _, L = x.shape  # [B, 2, L]

    # 合成复信号 z = I + jQ， shape [B, L]
    # 使用 torch.complex 构造，dtype 自动升级为对应复数类型
    z = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, L]  complex64/complex128

    # 分段 [B, n_seg, nperseg]
    frames = _frame_signal(z, nperseg=nperseg, noverlap=noverlap)  # [B, n_seg, nperseg]

    # 加窗（实窗作用于复信号）
    real_dtype = x.dtype  # 实数 dtype，用于窗函数
    win = _get_hann_window(nperseg, device=x.device, dtype=real_dtype)  # [nperseg]
    # 广播相乘: [B, n_seg, nperseg] * [nperseg] -> [B, n_seg, nperseg]
    frames_win = frames * win

    # FFT (复信号 -> 双边谱), 长度 nperseg
    # torch.fft.fft 对复输入返回完整双边谱 [B, n_seg, nperseg]
    spec = torch.fft.fft(frames_win, n=nperseg, dim=-1)  # [B, n_seg, nperseg]

    # 功率: |X|^2  -> [B, n_seg, nperseg]
    power = spec.real ** 2 + spec.imag ** 2

    # 段平均: [B, nperseg]
    power_mean = power.mean(dim=1)

    # 归一化因子: 1 / (fs * sum(w^2))
    win_norm = (win ** 2).sum()  # 标量
    psd_two_sided = power_mean / (fs * win_norm)  # [B, nperseg] 双边 PSD

    # 取单边: 长度 nperseg//2 + 1
    n_one = nperseg // 2 + 1
    psd = psd_two_sided[:, :n_one].clone()  # [B, n_one]

    # 单边谱: 除 DC(0) 和 Nyquist(n_one-1, 仅当 nperseg 偶数时存在) 外其余 *2
    # 这里假设 nperseg 为偶数（256 满足）
    if nperseg % 2 == 0:
        psd[:, 1:-1] = psd[:, 1:-1] * 2.0
    else:
        psd[:, 1:] = psd[:, 1:] * 2.0

    # 数值保护: 浮点误差可能产生极小负值
    psd = psd.clamp_min(0.0)

    return psd  # [B, n_one]


def welch_psd_log(
    x: torch.Tensor,
    nperseg: int = 256,
    noverlap: int = 128,
    fs: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Welch 法估计单边 PSD（log 域），全局主入口。

    Args:
        x: [B, 2, L] I/Q 双通道实信号
        nperseg / noverlap / fs: 同 welch_psd
        eps: log 数值保护，默认 1e-8

    Returns:
        log_psd: [B, nperseg // 2 + 1] log(PSD + eps)
    """
    psd = welch_psd(x, nperseg=nperseg, noverlap=noverlap, fs=fs)  # [B, n_one]
    return torch.log(psd + eps)  # [B, n_one]


# ---------------------------------------------------------------------------
# 简易自检（直接 python utils/psd.py 跑一下，确认 shape 与与 scipy 数值近似）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, L = 4, 1024
    x = torch.randn(B, 2, L)

    psd = welch_psd(x, nperseg=256, noverlap=128)
    log_psd = welch_psd_log(x, nperseg=256, noverlap=128)

    print(f"input shape         : {tuple(x.shape)}")
    print(f"psd shape           : {tuple(psd.shape)}   (期望 [{B}, 129])")
    print(f"log_psd shape       : {tuple(log_psd.shape)} (期望 [{B}, 129])")
    print(f"psd min / max       : {psd.min().item():.4e} / {psd.max().item():.4e}")
    print(f"log_psd min / max   : {log_psd.min().item():.4e} / {log_psd.max().item():.4e}")

    # 与 scipy 对照（白噪声理论 PSD ≈ 2 for 单边复信号方差归一化情况）
    try:
        import numpy as np
        from scipy.signal import welch as scipy_welch

        z_np = (x[:, 0, :] + 1j * x[:, 1, :]).numpy()  # [B, L]
        f_sp, psd_sp = scipy_welch(
            z_np[0],
            fs=1.0,
            window="hann",
            nperseg=256,
            noverlap=128,
            return_onesided=True,
            scaling="density",
        )
        psd_torch_0 = psd.numpy()[0]
        rel_err = np.abs(psd_sp - psd_torch_0) / (np.abs(psd_sp) + 1e-12)
        print(f"\n[scipy 对照] 第 0 样本最大相对误差: {rel_err.max():.3e}")
        print(f"[scipy 对照] 第 0 样本平均相对误差: {rel_err.mean():.3e}")
    except ImportError:
        print("\nscipy 未安装，跳过对照测试。")

    # GPU 测试
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        psd_gpu = welch_psd(x_gpu)
        print(f"\n[GPU] psd shape: {tuple(psd_gpu.shape)}, device: {psd_gpu.device}")
