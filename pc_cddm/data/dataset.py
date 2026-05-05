"""
PC-CDDM IQ 信号数据集。

H5 文件结构:
    filenames:   (N,) object  ← 含噪文件名，用于解析 SNR
    hz_signals:  (N, 2, 1024) float32  ← 含噪 IQ 信号 y
    wz_signals:  (N, 2, 1024) float32  ← 干净 IQ 信号 x_0

文件命名格式 (共 6 字段，下划线分隔，内部用连字符):
    [信号集]_[耦合类型]_[子类型]_[SNR]_[噪声]_[编号].mat

    SNR 字段 (index 3) 格式:
        "n15" -> -15.0 dB
        "n10" -> -10.0 dB
        "n5"  ->  -5.0 dB
        "p0"  ->   0.0 dB  (或 "0")
        "p5"  ->   5.0 dB
        "p10" ->  10.0 dB

设计要点:
    1. 训练时对 SNR 加 ±2dB 均匀扰动 (在 dataset __getitem__ 内完成)，
       验证时关闭扰动，保证 eval 可复现。
    2. 按文件顺序做 train/val 切分 (最后 val_ratio 部分做验证)，
       避免同编号样本同时出现在两个 split 中。
    3. preload=True 时把 hz/wz 全部读进 RAM (推荐 Kaggle GPU 训练)，
       preload=False 时保持 h5py lazy 读取 (适合内存受限场景)。
    4. __getitem__ 返回 (y, x0, snr_db):
       y [2, 1024] float32, x0 [2, 1024] float32, snr_db scalar float32
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

import h5py
import numpy as np


# ============================================================================
# SNR 字段解析
# ============================================================================
_SNR_PATTERN = re.compile(r"^([np]?)(\d+)$")


def parse_snr(snr_str: str) -> float:
    """
    将文件名中的 SNR 字段解析为物理 dB 值。

    规则:
        "n15" -> -15.0    (n 前缀表示负)
        "n5"  ->  -5.0
        "p5"  ->   5.0    (p 前缀表示正, 可省略)
        "p0"  ->   0.0
        "0"   ->   0.0    (纯数字视为正)
        "10"  ->  10.0

    Args:
        snr_str: SNR 字段字符串

    Returns:
        float: SNR dB 值

    Raises:
        ValueError: 无法解析时
    """
    m = _SNR_PATTERN.match(snr_str.strip())
    if m is None:
        raise ValueError(f"无法解析 SNR 字段: '{snr_str}'")
    sign_char, digits = m.group(1), m.group(2)
    value = float(digits)
    if sign_char == "n":
        value = -value
    return value


def parse_snr_from_filename(filename: str) -> float:
    """
    从完整文件名中提取 SNR。

    命名格式: [信号集]_[耦合类型]_[子类型]_[SNR]_[噪声]_[编号].mat
    SNR 字段固定在下划线分隔后的第 4 段 (index 3)。

    Args:
        filename: e.g. "HZ_LD-TX_LFM-QPSK_n10_GS-XW_00001.mat"

    Returns:
        float: SNR dB 值
    """
    # 取文件基名（去路径）
    basename = os.path.basename(filename)
    # 去掉 .mat 后缀再分割
    parts = basename.replace(".mat", "").split("_")
    if len(parts) < 6:
        raise ValueError(
            f"文件名字段数不足 6 段: '{filename}' -> {parts}"
        )
    snr_field = parts[3]  # index 3 固定为 SNR 字段
    return parse_snr(snr_field)


# ============================================================================
# IQDataset
# ============================================================================
class IQDataset(Dataset):
    """
    PC-CDDM IQ 信号数据集，从单个 H5 文件加载。

    Args:
        h5_path:        H5 文件路径
        split:          "train" | "val" | "all"
        val_ratio:      验证集比例 (按文件顺序切尾部), 默认 0.1
        snr_perturb_db: 训练时 SNR 扰动幅度 (±均匀), 默认 2.0 dB; val/all 模式为 0
        preload:        是否在 __init__ 时把 hz/wz 全部读进 RAM
        seed:           SNR 扰动用随机数种子基础值 (每个 __getitem__ 用 idx 为子种子)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        val_ratio: float = 0.1,
        snr_perturb_db: float = 2.0,
        preload: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        if split not in ("train", "val", "all"):
            raise ValueError(f"split 期望 'train'|'val'|'all', 实际 '{split}'")

        self.h5_path = h5_path
        self.split = split
        self.snr_perturb_db = snr_perturb_db if split == "train" else 0.0
        self.seed = seed

        # ------------------------------------------------------------------
        # 读取 H5 文件, 解析 SNR
        # ------------------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            N = f["hz_signals"].shape[0]

            # 解析 filenames -> SNR 数组 (全量, 再按 split 切)
            raw_filenames = f["filenames"][:]  # (N,) object / bytes
            snr_list: list[float] = []
            for fn in raw_filenames:
                # h5py 可能返回 bytes 或 str
                fn_str = fn.decode() if isinstance(fn, (bytes, bytearray)) else str(fn)
                snr_list.append(parse_snr_from_filename(fn_str))
            snr_all = np.array(snr_list, dtype=np.float32)  # (N,)

            # 按 split 切分索引 (按文件顺序, 尾部做 val)
            n_val = max(1, int(N * val_ratio))
            if split == "train":
                indices = np.arange(0, N - n_val)
            elif split == "val":
                indices = np.arange(N - n_val, N)
            else:  # "all"
                indices = np.arange(N)

            self.snr_db = torch.from_numpy(snr_all[indices])  # [M]

            # 按需 preload
            if preload:
                self.hz = torch.from_numpy(f["hz_signals"][indices])   # [M, 2, 1024]
                self.wz = torch.from_numpy(f["wz_signals"][indices])   # [M, 2, 1024]
                self._preloaded = True
            else:
                # lazy 模式: 存储索引, __getitem__ 时开文件读
                self._indices = indices
                self._preloaded = False
                self._h5_path = h5_path

        self._len = len(indices)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._len

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            y:      [2, 1024] float32  含噪 IQ 信号
            x0:     [2, 1024] float32  干净 IQ 信号
            snr_db: scalar float32     物理 SNR (dB), 训练时含 ±2dB 扰动
        """
        if self._preloaded:
            y  = self.hz[idx]    # [2, 1024]
            x0 = self.wz[idx]    # [2, 1024]
        else:
            # lazy 读取
            real_idx = int(self._indices[idx])
            with h5py.File(self._h5_path, "r") as f:
                y  = torch.from_numpy(f["hz_signals"][real_idx])  # [2, 1024]
                x0 = torch.from_numpy(f["wz_signals"][real_idx])  # [2, 1024]

        # SNR 扰动 (仅训练模式, snr_perturb_db > 0)
        snr_db = self.snr_db[idx].clone()  # 标量 float32
        if self.snr_perturb_db > 0:
            # 用 (seed + idx) 为每个样本生成确定性扰动 (多 worker 下可复现)
            rng = torch.Generator()
            rng.manual_seed(self.seed + idx)
            delta = (torch.rand(1, generator=rng).item() * 2 - 1) * self.snr_perturb_db
            snr_db = snr_db + delta

        return y, x0, snr_db

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        data_cfg: dict[str, Any],
        split: str = "train",
    ) -> "IQDataset":
        """
        从 yaml['data'] 段构造。

        期望字段:
            h5_path:        str
            val_ratio:      float (默认 0.1)
            snr_perturb_db: float (默认 2.0)
            preload:        bool  (默认 True)
            seed:           int   (默认 42)
        """
        return cls(
            h5_path=data_cfg["h5_path"],
            split=split,
            val_ratio=data_cfg.get("val_ratio", 0.1),
            snr_perturb_db=data_cfg.get("snr_perturb_db", 2.0),
            preload=data_cfg.get("preload", True),
            seed=data_cfg.get("seed", 42),
        )


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.data.dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    torch.manual_seed(0)

    # ==================================================================
    print("=" * 60)
    print("测试 1: SNR 字段解析")
    print("=" * 60)
    # ==================================================================
    cases = [
        ("n15", -15.0),
        ("n10", -10.0),
        ("n5",   -5.0),
        ("p0",    0.0),
        ("0",     0.0),
        ("p5",    5.0),
        ("p10",  10.0),
        ("10",   10.0),
    ]
    all_ok = True
    for s, expected in cases:
        got = parse_snr(s)
        ok = abs(got - expected) < 1e-6
        if not ok:
            all_ok = False
        print(f"  parse_snr('{s}') = {got:>6.1f}  (期望 {expected:>6.1f})  {'✓' if ok else '✗'}")
    print(f"SNR 解析全部正确: {'✓' if all_ok else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: 从文件名解析 SNR")
    print("=" * 60)
    # ==================================================================
    fn_cases = [
        ("HZ_DL_LFM_n15_GS_00001.mat",                  -15.0),
        ("HZ_LD-TX_LFM-QPSK_n10_GS-XW_00001.mat",       -10.0),
        ("HZ_LD-TX-GR_LFM-16QAM-ZB_n5_GS-FH-MCZ_00001.mat", -5.0),
        ("/some/path/HZ_DL_LFM_p5_GS_00099.mat",           5.0),
    ]
    all_ok2 = True
    for fn, expected in fn_cases:
        got = parse_snr_from_filename(fn)
        ok = abs(got - expected) < 1e-6
        if not ok:
            all_ok2 = False
        print(f"  {os.path.basename(fn)}")
        print(f"    -> {got:>6.1f} dB  (期望 {expected:>6.1f})  {'✓' if ok else '✗'}")
    print(f"文件名解析全部正确: {'✓' if all_ok2 else '✗'}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: 无效格式报错")
    print("=" * 60)
    # ==================================================================
    try:
        parse_snr("WZ")
        print("✗ 应抛出 ValueError 未抛出")
    except ValueError as e:
        print(f"✓ parse_snr('WZ') 正确报错: {e}")

    try:
        parse_snr_from_filename("WZ_DL_LFM_WZ_WZ_00001.mat")
        print("✗ 应抛出 ValueError 未抛出")
    except ValueError as e:
        print(f"✓ WZ 文件名正确报错: {e}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: IQDataset (mock H5 文件)")
    print("=" * 60)
    # ==================================================================
    # 构造 mock H5: 100 个样本, SNR 分布 n15/n10/n5/p0/p5/p10 各若干
    N_mock = 100
    snr_vals = [-15, -10, -5, 0, 5, 10]

    def _snr_to_str(v: int) -> str:
        if v < 0:
            return f"n{abs(v)}"
        return f"p{v}"

    mock_filenames = []
    for i in range(N_mock):
        snr_v = snr_vals[i % len(snr_vals)]
        fn = f"HZ_DL_LFM_{_snr_to_str(snr_v)}_GS_{i+1:05d}.mat"
        mock_filenames.append(fn.encode())

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("filenames",  data=np.array(mock_filenames, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
        f.create_dataset("hz_signals", data=np.random.randn(N_mock, 2, 1024).astype(np.float32))
        f.create_dataset("wz_signals", data=np.random.randn(N_mock, 2, 1024).astype(np.float32))

    # ---- train split ----
    ds_train = IQDataset(tmp_path, split="train", val_ratio=0.1,
                         snr_perturb_db=2.0, preload=True)
    ds_val   = IQDataset(tmp_path, split="val",   val_ratio=0.1,
                         snr_perturb_db=2.0, preload=True)

    n_val_expected = max(1, int(N_mock * 0.1))
    n_train_expected = N_mock - n_val_expected

    print(f"总样本: {N_mock}, train: {len(ds_train)}, val: {len(ds_val)}")
    print(f"train 长度正确: {'✓' if len(ds_train) == n_train_expected else '✗'}")
    print(f"val   长度正确: {'✓' if len(ds_val)   == n_val_expected else '✗'}")
    print(f"train + val = 总: {'✓' if len(ds_train) + len(ds_val) == N_mock else '✗'}")

    # ---- __getitem__ shape / dtype ----
    y, x0, snr = ds_train[0]
    print(f"\n__getitem__ 返回:")
    print(f"  y   shape={tuple(y.shape)}  dtype={y.dtype}   (期望 [2,1024] float32)")
    print(f"  x0  shape={tuple(x0.shape)} dtype={x0.dtype}  (期望 [2,1024] float32)")
    print(f"  snr shape={tuple(snr.shape) if snr.dim()>0 else 'scalar'}  dtype={snr.dtype}  值={snr.item():.2f}")
    print(f"  shape ✓: {y.shape == (2, 1024) and x0.shape == (2, 1024)}")
    print(f"  dtype ✓: {y.dtype == torch.float32}")

    # ---- SNR 扰动: 同一 idx 每次调用结果一致 ----
    _, _, snr_a = ds_train[5]
    _, _, snr_b = ds_train[5]
    print(f"\nSNR 扰动可复现 (idx=5 两次结果相同): {'✓' if snr_a.item() == snr_b.item() else '✗'}")
    print(f"  SNR idx=5: {snr_a.item():.3f} dB (未扰动应为 {ds_train.snr_db[5].item():.1f} dB)")

    # ---- 验证集无扰动 ----
    _, _, snr_val_raw = ds_val[0]
    _, _, snr_val_raw2 = ds_val[0]
    print(f"验证集无扰动 (snr_perturb_db 被置 0): "
          f"{'✓' if ds_val.snr_perturb_db == 0.0 else '✗'}")

    # ---- SNR 范围合理性 ----
    all_snrs = ds_train.snr_db
    print(f"\ntrain SNR 范围: [{all_snrs.min().item():.1f}, {all_snrs.max().item():.1f}] dB")
    in_range = (all_snrs >= -15.0 - 1e-3) & (all_snrs <= 10.0 + 1e-3)
    print(f"SNR 全部在 [-15, 10] 内: {'✓' if in_range.all() else '✗'}")

    # ---- DataLoader 兼容 ----
    from torch.utils.data import DataLoader
    loader = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=0)
    y_batch, x0_batch, snr_batch = next(iter(loader))
    print(f"\nDataLoader batch shape:")
    print(f"  y_batch   {tuple(y_batch.shape)}   (期望 [8, 2, 1024])")
    print(f"  x0_batch  {tuple(x0_batch.shape)}  (期望 [8, 2, 1024])")
    print(f"  snr_batch {tuple(snr_batch.shape)} (期望 [8])")
    print(f"  ✓: {y_batch.shape == (8, 2, 1024) and snr_batch.shape == (8,)}")

    # ---- lazy 模式 ----
    print("\n" + "=" * 60)
    print("测试 5: preload=False lazy 模式")
    print("=" * 60)
    ds_lazy = IQDataset(tmp_path, split="train", val_ratio=0.1,
                        snr_perturb_db=2.0, preload=False)
    y_l, x0_l, snr_l = ds_lazy[0]
    print(f"lazy __getitem__ shape ✓: {y_l.shape == (2, 1024)}")
    print(f"lazy NaN 检查: {'✓ 无 NaN' if not y_l.isnan().any() else '✗'}")

    # ---- from_config ----
    print("\n" + "=" * 60)
    print("测试 6: from_config")
    print("=" * 60)
    fake_data_cfg = {
        "h5_path": tmp_path,
        "val_ratio": 0.1,
        "snr_perturb_db": 2.0,
        "preload": True,
        "seed": 42,
    }
    ds_cfg = IQDataset.from_config(fake_data_cfg, split="train")
    print(f"from_config 构造成功, len={len(ds_cfg)}  {'✓' if len(ds_cfg) == len(ds_train) else '✗'}")

    # 清理临时文件
    os.unlink(tmp_path)

    print("\n" + "=" * 60)
    print("全部自检完成")
    print("=" * 60)
