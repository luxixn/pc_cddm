"""
PSD estimation via Welch's method, GPU-native PyTorch implementation.

设计要点：
1. 输入为 [B, 2, L] 的 I/Q 双通道实信号，内部合成复信号 z = I + jQ。
2. 输出为 **双边谱 [B, nperseg]**（默认 nperseg=256 -> 256 维）。
   双边谱对复信号是物理正确的表示——正负频独立、无信息损失。
3. 频率 bin 顺序遵循 torch.fft.fft / scipy fftfreq 自然约定：
        index 0          ->  DC (f = 0)
        index 1..N/2-1   ->  正频 +1·fs/N, ..., +(N/2-1)·fs/N
        index N/2        ->  Nyquist (±fs/2)
        index N/2+1..N-1 ->  负频 -(N/2-1)·fs/N, ..., -1·fs/N
4. log 域接口为全局主入口；原始 PSD 接口保留以备消融。
5. 使用 torch.fft.fft 在 GPU 上执行，避免 CPU 拷贝。

数学约定（与 scipy.signal.welch(..., return_onesided=False, scaling='density') 一致）:
    - 加 Hann 窗 w[n]
    - 每段 segment 做 FFT 后取 |X|^2
    - 归一化因子: 1 / (fs * sum(w^2))
    - 不做单边折叠，直接返回完整双边谱
    - 多段平均得到最终 PSD

注意: 这里采用归一化频率 fs=1.0。如需物理频率可在调用方传 fs。
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Hann 窗缓存：避免每次调用都重新构造
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
    detrend: bool = True,
) -> torch.Tensor:
    """
    Welch 法估计双边 PSD（线性域）。

    输入 IQ 双通道实信号 [B, 2, L]，内部合成复信号 z = I + jQ，
    输出复信号的双边功率谱密度 [B, nperseg]。

    Args:
        x: [B, 2, L] I/Q 双通道实信号 (channel 0 = I, channel 1 = Q)
        nperseg: 每段长度，默认 256
        noverlap: 段间重叠样本数，默认 128
        fs: 采样率，归一化频率下取 1.0
        detrend: 是否对每段做常数去趋势（减段均值），默认 True，
                 与 scipy.signal.welch 默认行为(detrend='constant') 一致

    Returns:
        psd: [B, nperseg] 双边 PSD（线性域，非负）
             频率排布: [0, +f1..+f_{N/2-1}, ±Nyquist, -f_{N/2-1}..-f1]
    """
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"期望输入 shape [B, 2, L]，实际 {tuple(x.shape)}")

    B, _, L = x.shape  # [B, 2, L]

    # 合成复信号 z = I + jQ， shape [B, L]
    # torch.complex 自动从实数 dtype 升级为对应复数 dtype（float32 -> complex64）
    z = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, L]

    # 分段 [B, n_seg, nperseg]
    frames = _frame_signal(z, nperseg=nperseg, noverlap=noverlap)  # [B, n_seg, nperseg]

    # 常数去趋势：每段减去段均值（对齐 scipy detrend='constant'）
    # 主要影响 DC bin，对其他 bin 几乎无影响。但加上更标准、可复现性强。
    if detrend:
        frames = frames - frames.mean(dim=-1, keepdim=True)  # [B, n_seg, nperseg]

    # 加窗（实窗作用于复信号，逐元素相乘）
    real_dtype = x.dtype  # 实数 dtype，用于窗函数
    win = _get_hann_window(nperseg, device=x.device, dtype=real_dtype)  # [nperseg]
    # 广播相乘: [B, n_seg, nperseg] * [nperseg] -> [B, n_seg, nperseg]
    frames_win = frames * win

    # FFT (复信号 -> 双边谱), 长度 nperseg
    spec = torch.fft.fft(frames_win, n=nperseg, dim=-1)  # [B, n_seg, nperseg]

    # 功率: |X|^2 -> [B, n_seg, nperseg]
    power = spec.real ** 2 + spec.imag ** 2

    # 段平均: [B, nperseg]
    power_mean = power.mean(dim=1)

    # 归一化因子: 1 / (fs * sum(w^2))，得到 PSD（功率谱密度）
    win_norm = (win ** 2).sum()  # 标量
    psd = power_mean / (fs * win_norm)  # [B, nperseg]

    # 数值保护: 浮点误差可能产生极小负值
    psd = psd.clamp_min(0.0)

    return psd  # [B, nperseg]


def welch_psd_log(
    x: torch.Tensor,
    nperseg: int = 256,
    noverlap: int = 128,
    fs: float = 1.0,
    eps: float = 1e-8,
    detrend: bool = True,
) -> torch.Tensor:
    """
    Welch 法估计双边 PSD（log 域），全局主入口。

    Args:
        x: [B, 2, L] I/Q 双通道实信号
        nperseg / noverlap / fs / detrend: 同 welch_psd
        eps: log 数值保护，默认 1e-8

    Returns:
        log_psd: [B, nperseg] log(PSD + eps)
    """
    psd = welch_psd(x, nperseg=nperseg, noverlap=noverlap, fs=fs, detrend=detrend)
    return torch.log(psd + eps)  # [B, nperseg]


# ---------------------------------------------------------------------------
# 自检：python -m pc_cddm.utils.psd
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, L = 4, 1024
    x = torch.randn(B, 2, L)  # [B, 2, L]，I/Q 各 ~ N(0, 1)

    psd = welch_psd(x, nperseg=256, noverlap=128)
    log_psd = welch_psd_log(x, nperseg=256, noverlap=128)

    print(f"input shape         : {tuple(x.shape)}")
    print(f"psd shape           : {tuple(psd.shape)}   (期望 [{B}, 256])")
    print(f"log_psd shape       : {tuple(log_psd.shape)} (期望 [{B}, 256])")
    print(f"psd min / max       : {psd.min().item():.4e} / {psd.max().item():.4e}")
    print(f"log_psd min / max   : {log_psd.min().item():.4e} / {log_psd.max().item():.4e}")

    # ----- 与 scipy 严格逐元素对照 -----
    # scipy 对复信号自动用双边谱，与我们一致，可直接逐 bin 比较
    try:
        import warnings
        import numpy as np
        from scipy.signal import welch as scipy_welch

        z_np = (x[:, 0, :] + 1j * x[:, 1, :]).numpy()  # [B, L]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 屏蔽 scipy 复信号 warning
            f_sp, psd_sp = scipy_welch(
                z_np[0],
                fs=1.0,
                window="hann",
                nperseg=256,
                noverlap=128,
                return_onesided=False,
                scaling="density",
            )
        # scipy return_onesided=False 返回的频率顺序与 torch.fft.fft 一致
        # （[0, +f, ..., -f]），可直接逐元素比较
        psd_torch_0 = psd.numpy()[0]  # [256]
        rel_err = np.abs(psd_sp - psd_torch_0) / (np.abs(psd_sp) + 1e-12)
        print(f"\n[scipy 对照] shape: scipy={psd_sp.shape}, torch={psd_torch_0.shape}")
        print(f"[scipy 对照] 最大相对误差: {rel_err.max():.3e}")
        print(f"[scipy 对照] 平均相对误差: {rel_err.mean():.3e}")
        print(f"[scipy 对照] 期望: < 1e-5 (已对齐 scipy 默认 detrend='constant')")
    except ImportError:
        print("\nscipy 未安装，跳过对照测试。")

    # ----- 白噪声理论值健全性检查 -----
    # 复白噪声 z = I + jQ, I,Q ~ N(0,1) 独立 -> Var(z) = 2
    # 双边 PSD 在 [0, fs) 上的积分 = Var(z) = 2
    # 平均 PSD = Var(z) / fs = 2.0  (fs=1.0)
    print(f"\n[白噪声理论] 期望双边 PSD 平均值 ≈ 2.0")
    print(f"[白噪声理论] 实际 PSD 平均值        = {psd.mean().item():.4f}")
    # 单一批次会有抖动，多批次平均会更准
    psd_many = welch_psd(torch.randn(64, 2, L))
    print(f"[白噪声理论] 64 批次平均后          = {psd_many.mean().item():.4f}")

    # ----- GPU 测试 -----
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        psd_gpu = welch_psd(x_gpu)
        print(f"\n[GPU] psd shape: {tuple(psd_gpu.shape)}, device: {psd_gpu.device}")
