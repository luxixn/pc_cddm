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


def parse_snr_from_filename(filename: str) -> float | None:
    """
    从文件名 (可能是组合格式) 中提取 SNR。

    支持的输入格式:
        1. 组合格式 (实际数据集使用):
           "WZ:WZ_DL_LFM_WZ_WZ_00001.mat | HZ:HZ_DL_LFM_n15_GS_00001.mat"
           -> 提取 HZ 段, 解析 SNR
        2. 纯 HZ 文件名:
           "HZ_DL_LFM_n15_GS_00001.mat"
           -> 直接解析 SNR
        3. 纯 WZ 无噪样本 (未配对):
           "WZ_DL_LFM_WZ_WZ_00001.mat"
           -> 返回 None, 调用方 (IQDataset) 会跳过
           (实际数据集每一行都是组合格式, 此分支仅作防御)

    命名规则 (HZ 部分):
        [HZ]_[耦合类型]_[子类型]_[SNR]_[噪声]_[编号].mat
        SNR 字段在第 4 段 (index 3), 例:
            "n15"  -> -15 dB
            "n10"  -> -10 dB
            "n5"   ->  -5 dB
            "0"    ->   0 dB   (无 p 前缀)
            "5"    ->   5 dB
            "10"   ->  10 dB

    Args:
        filename: 完整文件名字符串

    Returns:
        float: SNR dB 值; None 表示无噪样本
    """
    s = filename.strip()

    # 1. 组合格式 "WZ:... | HZ:..." -> 提取 HZ 段
    if "HZ:" in s:
        idx = s.find("HZ:")
        hz_part = s[idx + 3:]               # 去掉 "HZ:" 前缀
        if "|" in hz_part:                  # 防御: HZ 后面再有竖线就截断
            hz_part = hz_part.split("|", 1)[0]
        s = hz_part.strip()
    elif s.startswith("WZ:"):
        # 只剩 WZ 段 (没有配对 HZ): 视作无噪样本
        return None

    # 2. 走标准字段切分
    basename = os.path.basename(s)
    parts = basename.replace(".mat", "").split("_")
    if len(parts) < 6:
        raise ValueError(
            f"文件名字段数不足 6 段: '{filename}' -> {parts}"
        )
    snr_field = parts[3]
    if snr_field == "WZ":
        return None                         # 防御: 纯 WZ 文件名也支持
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
        # 读取 H5 文件, 解析 SNR, 过滤 WZ 无噪样本
        # ------------------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            N_total = f["hz_signals"].shape[0]

            # 解析 filenames -> SNR 数组, 同时收集 HZ 样本的真实索引
            raw_filenames = f["filenames"][:]  # (N_total,) object / bytes
            snr_list: list[float] = []
            valid_idx: list[int] = []          # 在 H5 内的真实行号
            n_wz_skipped = 0
            for i, fn in enumerate(raw_filenames):
                fn_str = fn.decode() if isinstance(fn, (bytes, bytearray)) else str(fn)
                snr_val = parse_snr_from_filename(fn_str)
                if snr_val is None:
                    # WZ 无噪样本: 跳过 (训练去噪模型用不到)
                    n_wz_skipped += 1
                    continue
                snr_list.append(snr_val)
                valid_idx.append(i)

            if n_wz_skipped > 0:
                print(
                    f"[IQDataset] 跳过 {n_wz_skipped} 个 WZ 无噪样本 "
                    f"(剩余 {len(valid_idx)} / {N_total} 个 HZ 含噪样本)"
                )

            if len(valid_idx) == 0:
                raise RuntimeError(
                    f"H5 中没有任何 HZ 含噪样本 (跳过 {n_wz_skipped} 个 WZ); "
                    f"请检查数据文件 {h5_path}"
                )

            valid_idx_arr = np.array(valid_idx, dtype=np.int64)  # H5 行号
            snr_all = np.array(snr_list, dtype=np.float32)        # 与 valid_idx 对齐
            N = len(valid_idx)                                    # 有效样本数

            # 按 split 切分 (在过滤后样本上做切分)
            n_val = max(1, int(N * val_ratio))
            if split == "train":
                sub = np.arange(0, N - n_val)
            elif split == "val":
                sub = np.arange(N - n_val, N)
            else:  # "all"
                sub = np.arange(N)

            self.snr_db = torch.from_numpy(snr_all[sub])  # [M]

            # sub 索引映射回 H5 真实行号 (用于 hz/wz 读取)
            real_indices = valid_idx_arr[sub]             # H5 行号, 单调递增

            # 按需 preload
            if preload:
                # h5py 支持 ndarray 索引 (递增); valid_idx 按 i 顺序构建天然递增
                self.hz = torch.from_numpy(f["hz_signals"][real_indices])  # [M, 2, L]
                self.wz = torch.from_numpy(f["wz_signals"][real_indices])  # [M, 2, L]
                self._preloaded = True
            else:
                # lazy 模式: 存储 H5 真实行号
                self._indices = real_indices
                self._preloaded = False
                self._h5_path = h5_path

        self._len = len(sub)

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
        # 实际数据集的组合格式 (WZ:... | HZ:...)
        ("WZ:WZ_DL_LFM_WZ_WZ_00001.mat | HZ:HZ_DL_LFM_n15_GS_00001.mat", -15.0),
        ("WZ:WZ_TX-GR_QPSK-MC_WZ_WZ_29926.mat | HZ:HZ_TX-GR_QPSK-MC_0_GS_29926.mat", 0.0),
        ("WZ:WZ_LD-TX-GR_LFM-16QAM-MC_WZ_WZ_59850.mat | HZ:HZ_LD-TX-GR_LFM-16QAM-MC_10_GS-FH-MCZ_59850.mat", 10.0),
        ("WZ:WZ_LD-TX-GR_LFM-QPSK-KD_WZ_WZ_44888.mat | HZ:HZ_LD-TX-GR_LFM-QPSK-KD_5_GS-FH_44888.mat", 5.0),
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

    # WZ 文件名: 现在返回 None (而不是报错), 由 IQDataset 跳过
    wz_result = parse_snr_from_filename("WZ_DL_LFM_WZ_WZ_00001.mat")
    print(f"✓ WZ 文件名返回 None (无噪样本): {wz_result is None}")

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

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 7: 混合 WZ + HZ 数据集, 验证 WZ 自动过滤")
    print("=" * 60)
    # ==================================================================
    # 构造: 50 HZ + 30 WZ 共 80 个样本
    N_hz, N_wz = 50, 30
    N_mix = N_hz + N_wz
    mock_filenames_mix = []
    # 先 HZ 后 WZ (检查跳过逻辑不依赖位置)
    for i in range(N_hz):
        snr_v = snr_vals[i % len(snr_vals)]
        mock_filenames_mix.append(
            f"HZ_DL_LFM_{_snr_to_str(snr_v)}_GS_{i+1:05d}.mat".encode()
        )
    for i in range(N_wz):
        mock_filenames_mix.append(
            f"WZ_DL_LFM_WZ_WZ_{i+1:05d}.mat".encode()
        )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path_mix = tmp.name
    with h5py.File(tmp_path_mix, "w") as f:
        f.create_dataset("filenames",  data=np.array(mock_filenames_mix, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))
        f.create_dataset("hz_signals", data=np.random.randn(N_mix, 2, 1024).astype(np.float32))
        f.create_dataset("wz_signals", data=np.random.randn(N_mix, 2, 1024).astype(np.float32))

    print("(下面这一行 [IQDataset] 输出是预期的:)")
    ds_mix = IQDataset(tmp_path_mix, split="all", val_ratio=0.1,
                       snr_perturb_db=0.0, preload=True)
    print(f"\n过滤后样本数: {len(ds_mix)}  (期望 {N_hz}, 即只保留 HZ)")
    print(f"WZ 全部跳过 ✓: {len(ds_mix) == N_hz}")

    # 验证保留的 SNR 全部在合理范围
    in_range = (ds_mix.snr_db >= -15.0) & (ds_mix.snr_db <= 10.0)
    print(f"剩余 SNR 全在 [-15, 10] 内 ✓: {in_range.all().item()}")

    # __getitem__ 不应再触发任何 ValueError
    y, x0, snr = ds_mix[0]
    print(f"__getitem__[0] 正常返回 ✓: shape={tuple(y.shape)}, snr={snr.item():.1f}")

    os.unlink(tmp_path_mix)

    print("\n" + "=" * 60)
    print("全部自检完成")
    print("=" * 60)
