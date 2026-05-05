"""
PC-CDDM 训练日志与 checkpoint 管理。

包含:
    setup_run_dir       创建 runs/<exp_name>/ 子目录树, 返回路径字典
    Logger              同时输出到 stdout 与文件的带时间戳日志器
    CheckpointManager   ckpt 保存/加载/清理 (含 latest / best 维护)

设计要点:
    1. Logger 不依赖 logging 标准库的复杂配置, 直接 print + open(file, 'a'),
       保证 Kaggle 多进程/重启后日志文件可追加而不丢。
    2. CheckpointManager 在 Windows 不依赖软链接, 用复制实现 ckpt_latest.pt
       的"指向最新"语义, 保证跨平台一致。
    3. ckpt 文件名带 epoch 编号便于人眼对照; 自动清理时跳过 best 与 latest。
    4. 所有路径用 pathlib.Path, 跨平台分隔符无需手动处理。
"""

from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


# ============================================================================
# Run 目录树
# ============================================================================
def setup_run_dir(
    output_root: str | Path,
    exp_name: str,
    *,
    create: bool = True,
) -> dict[str, Path]:
    """
    建立 runs/<exp_name>/ 子目录, 返回路径字典。

    目录结构:
        <output_root>/<exp_name>/
            log.txt          训练日志
            ckpts/           checkpoint 目录
            tb/              tensorboard 目录 (可选, 训练循环按需创建 SummaryWriter)
            config.yaml      训练时把配置 dump 进来 (train.py 负责)

    Args:
        output_root: 输出根目录, e.g. "runs/" 或 "/kaggle/working/runs/"
        exp_name:    实验名, e.g. "exp_default"
        create:      True 时创建所有子目录

    Returns:
        dict, 含:
            "run_dir":  Path  运行根目录
            "log_file": Path  日志文件
            "ckpt_dir": Path  checkpoint 目录
            "tb_dir":   Path  tensorboard 目录
            "config_file": Path  配置存档
    """
    run_dir  = Path(output_root) / exp_name
    log_file = run_dir / "log.txt"
    ckpt_dir = run_dir / "ckpts"
    tb_dir   = run_dir / "tb"
    cfg_file = run_dir / "config.yaml"

    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir":     run_dir,
        "log_file":    log_file,
        "ckpt_dir":    ckpt_dir,
        "tb_dir":      tb_dir,
        "config_file": cfg_file,
    }


# ============================================================================
# Logger
# ============================================================================
class Logger:
    """
    极简日志器: 同时输出到 stdout 和文件, 自动带时间戳。

    使用方式:
        logger = Logger("runs/exp1/log.txt")
        logger.info("训练开始")
        logger.info(f"epoch={ep} loss={loss:.4f}")
        logger.close()  # 或用 with 上下文管理器

    Args:
        log_file:  日志文件路径; None 时只输出到 stdout
        also_print: 是否同时打印到 stdout (默认 True)
        flush:     是否每次 info 都 flush 文件 (默认 True, 防 Kaggle 中断丢日志)
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        *,
        also_print: bool = True,
        flush: bool = True,
    ):
        self.log_file = Path(log_file) if log_file is not None else None
        self.also_print = also_print
        self.flush_each = flush
        self._fh = None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # append 模式, 续训时不覆盖
            self._fh = self.log_file.open("a", encoding="utf-8")

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str) -> None:
        line = f"[{self._ts()}] {msg}"
        if self.also_print:
            print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            if self.flush_each:
                self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ============================================================================
# CheckpointManager
# ============================================================================
class CheckpointManager:
    """
    Checkpoint 保存/加载/清理。

    文件命名:
        ckpt_epoch{N:04d}.pt    带 epoch 编号的常规 ckpt
        ckpt_latest.pt          最新一个的副本 (续训默认入口)
        ckpt_best.pt            验证集最佳的副本 (eval 默认入口)

    自动清理:
        除 latest / best 外, 仅保留最近 keep_last_n 个 epoch ckpt。
        (best 即便 epoch 很旧也永不删; latest 始终覆盖为最新)

    Args:
        ckpt_dir:    checkpoint 保存目录
        keep_last_n: 保留多少个最近的 epoch ckpt (默认 3)
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        keep_last_n: int = 3,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    # ------------------------------------------------------------------
    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """
        保存 state 到 ckpt_epoch{N}.pt, 同步更新 latest, 必要时更新 best。

        Args:
            state:   要保存的状态字典 (典型含 model_state, optimizer_state, epoch, step, ...)
            epoch:   当前 epoch (用于命名)
            is_best: 是否同时更新 ckpt_best.pt

        Returns:
            主 ckpt 文件路径 (ckpt_epoch{N}.pt)
        """
        # 主 ckpt 文件
        ckpt_path = self.ckpt_dir / f"ckpt_epoch{epoch:04d}.pt"
        torch.save(state, ckpt_path)

        # 同步 latest (复制实现, 跨平台一致)
        latest_path = self.ckpt_dir / "ckpt_latest.pt"
        shutil.copyfile(ckpt_path, latest_path)

        # 必要时更新 best
        if is_best:
            best_path = self.ckpt_dir / "ckpt_best.pt"
            shutil.copyfile(ckpt_path, best_path)

        # 自动清理超额的旧 ckpt
        self._cleanup()

        return ckpt_path

    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """
        删除超出 keep_last_n 的旧 ckpt_epoch{N}.pt 文件 (latest/best 不动)。
        """
        epoch_ckpts = sorted(self.ckpt_dir.glob("ckpt_epoch*.pt"))
        # 保留最新 keep_last_n 个 (按 epoch 编号排序, 字典序与数值序一致因 04d)
        if len(epoch_ckpts) > self.keep_last_n:
            for old in epoch_ckpts[: -self.keep_last_n]:
                try:
                    old.unlink()
                except OSError:
                    pass  # 文件被占用等情况静默跳过, 不阻塞训练

    # ------------------------------------------------------------------
    def load(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """
        从给定路径加载 ckpt。

        Args:
            path:         ckpt 文件路径
            map_location: torch.load 参数, e.g. 'cpu' / 'cuda'

        Returns:
            state dict
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ckpt 不存在: {path}")
        # weights_only=False: 我们存的是完整训练状态 (含 optimizer 等), 必须 False
        return torch.load(path, map_location=map_location, weights_only=False)

    # ------------------------------------------------------------------
    def find_latest(self) -> Optional[Path]:
        """
        返回 ckpt_latest.pt 路径; 若不存在尝试 fallback 到最新 epoch ckpt; 都没有返回 None。
        """
        latest = self.ckpt_dir / "ckpt_latest.pt"
        if latest.exists():
            return latest
        # fallback: 最大 epoch 编号的常规 ckpt
        epoch_ckpts = sorted(self.ckpt_dir.glob("ckpt_epoch*.pt"))
        return epoch_ckpts[-1] if epoch_ckpts else None

    # ------------------------------------------------------------------
    def find_best(self) -> Optional[Path]:
        best = self.ckpt_dir / "ckpt_best.pt"
        return best if best.exists() else None


# ============================================================================
# Wallclock 计时器 (辅助会话感知退出)
# ============================================================================
class WallclockTimer:
    """
    简单的墙钟计时器, 训练循环用来判断是否接近 Kaggle 9 小时上限。

    用法:
        timer = WallclockTimer(max_hours=8.0)
        ...
        if timer.exceeded():
            logger.info("接近会话上限, 提前退出并保存")
            break

    Args:
        max_hours: 上限小时数 (达到后 exceeded() 返回 True)
    """

    def __init__(self, max_hours: float = 8.0):
        self.max_seconds = max_hours * 3600.0
        self.start_time = time.time()

    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600.0

    def exceeded(self) -> bool:
        return (time.time() - self.start_time) >= self.max_seconds

    def remaining_hours(self) -> float:
        return max(0.0, (self.max_seconds - (time.time() - self.start_time)) / 3600.0)


# ---------------------------------------------------------------------------
# 自检: python -m pc_cddm.utils.logging
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # ==================================================================
    print("=" * 60)
    print("测试 1: setup_run_dir 创建目录树")
    print("=" * 60)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = setup_run_dir(tmpdir, "exp_test")
        for k, v in paths.items():
            print(f"  {k:12s}: {v}")
        # 验证目录确实创建
        assert paths["run_dir"].is_dir(),  "run_dir 未创建"
        assert paths["ckpt_dir"].is_dir(), "ckpt_dir 未创建"
        assert paths["tb_dir"].is_dir(),   "tb_dir 未创建"
        # log_file / config_file 是文件路径, 不应自动创建
        assert not paths["log_file"].exists(),    "log_file 不该被预创建"
        assert not paths["config_file"].exists(), "config_file 不该被预创建"
        print("  目录创建 ✓, 文件路径未预创建 ✓")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 2: Logger 写文件 + stdout, 带时间戳, append 模式")
    print("=" * 60)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "log.txt"

        # 第一次会话
        logger = Logger(log_path, also_print=False)  # 关 stdout 避免污染自检输出
        logger.info("session-1 start")
        logger.info("session-1 step 100 loss 0.5")
        logger.close()

        # 第二次会话 (模拟续训)
        logger = Logger(log_path, also_print=False)
        logger.info("session-2 resume")
        logger.close()

        content = log_path.read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if l.strip()]
        print(f"  日志总行数: {len(lines)}  (期望 3)")
        for ln in lines:
            print(f"    {ln}")
        # 验证带时间戳格式
        import re
        ts_re = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")
        all_ts = all(ts_re.match(ln) for ln in lines)
        print(f"  全部带时间戳前缀 ✓: {all_ts}")
        print(f"  append 模式 (3 行 = 2+1) ✓: {len(lines) == 3}")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 3: Logger 上下文管理器")
    print("=" * 60)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "log.txt"
        with Logger(log_path, also_print=False) as logger:
            logger.info("ctx 1")
            logger.info("ctx 2")
        # 退出 with 后文件应已关闭, 内容完整
        content = log_path.read_text(encoding="utf-8")
        assert "ctx 1" in content and "ctx 2" in content
        print("  上下文管理器自动 close ✓")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 4: CheckpointManager 保存与加载")
    print("=" * 60)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(Path(tmpdir) / "ckpts", keep_last_n=3)

        # 模拟训练状态
        fake_state = {
            "epoch": 1,
            "step": 100,
            "model_state": {"weight": torch.randn(4, 4)},
            "optimizer_state": {"lr": 1e-4},
        }

        # 保存 5 个 epoch
        for ep in range(1, 6):
            fake_state["epoch"] = ep
            is_best = (ep == 3)  # 假设 epoch 3 最优
            cm.save(fake_state, epoch=ep, is_best=is_best)

        # 列文件
        files = sorted(p.name for p in (Path(tmpdir) / "ckpts").iterdir())
        print("  ckpt 目录文件:")
        for f in files:
            print(f"    {f}")

        # 验证: 应保留 epoch3,4,5 (keep_last_n=3) + latest + best
        # epoch 1, 2 应被清理
        epoch_files = [f for f in files if f.startswith("ckpt_epoch")]
        print(f"  epoch ckpt 数: {len(epoch_files)}  (期望 3)")
        print(f"  latest 存在 ✓: {'ckpt_latest.pt' in files}")
        print(f"  best 存在 ✓:   {'ckpt_best.pt' in files}")
        print(f"  epoch1 已清理 ✓: {'ckpt_epoch0001.pt' not in files}")
        print(f"  epoch3 (best) 仍在 ✓: {'ckpt_epoch0003.pt' in files}")

        # 加载验证
        latest_path = cm.find_latest()
        loaded = cm.load(latest_path)
        print(f"\n  加载 latest: epoch={loaded['epoch']} (期望 5)")
        print(f"  字段完整 ✓: {set(loaded.keys()) == {'epoch', 'step', 'model_state', 'optimizer_state'}}")

        best_path = cm.find_best()
        loaded_best = cm.load(best_path)
        print(f"  加载 best:   epoch={loaded_best['epoch']} (期望 3)")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 5: find_latest 在空目录返回 None")
    print("=" * 60)
    # ==================================================================
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(Path(tmpdir) / "ckpts")
        assert cm.find_latest() is None, "空目录应返回 None"
        assert cm.find_best() is None,   "空目录应返回 None"
        print("  空目录 find_latest=None ✓, find_best=None ✓")

    # ==================================================================
    print("\n" + "=" * 60)
    print("测试 6: WallclockTimer")
    print("=" * 60)
    # ==================================================================
    timer = WallclockTimer(max_hours=1.0 / 3600.0)  # 1 秒上限
    elapsed_0 = timer.elapsed_hours()
    print(f"  起始 elapsed_hours = {elapsed_0:.6f}  (期望 ~0)")
    print(f"  起始 exceeded = {timer.exceeded()}  (期望 False)")

    time.sleep(1.2)
    elapsed_1 = timer.elapsed_hours()
    print(f"  1.2s 后 elapsed_hours = {elapsed_1:.6f}  (期望 > 0)")
    print(f"  1.2s 后 exceeded = {timer.exceeded()}  (期望 True)")
    print(f"  remaining_hours = {timer.remaining_hours():.6f}  (期望 0)")
    print(f"  ✓: {timer.exceeded() and timer.remaining_hours() == 0.0}")

    print("\n" + "=" * 60)
    print("全部自检完成")
    print("=" * 60)
