"""
检查点管理模块 / Checkpoint Management Module

提供模型检查点的保存、加载和管理功能，支持断点续训。
Provides model checkpoint save, load, and management with resume training support.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("image_classifier")


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str,
    is_best: bool = False,
    best_filename: str = "best_model.pth",
) -> str:
    """
    保存训练检查点。
    Save training checkpoint.

    检查点包含内容 / Checkpoint contents:
        - epoch: 当前训练轮次
        - model_state_dict: 模型权重
        - optimizer_state_dict: 优化器状态（用于断点续训）
        - scheduler_state_dict: 学习率调度器状态
        - best_acc: 历史最佳验证准确率
        - config: 训练配置

    Args:
        state (dict): 要保存的状态字典。State dictionary to save.
        save_dir (str): 保存目录。Save directory.
        filename (str): 检查点文件名。Checkpoint filename.
        is_best (bool): 是否为最佳模型（同时保存为 best_model.pth）。
                        Whether this is the best model.
        best_filename (str): 最佳模型文件名。Best model filename.

    Returns:
        str: 保存的检查点文件路径。Saved checkpoint file path.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # 若为最佳模型，额外保存一份
    # If best model, save an additional copy
    if is_best:
        best_path = save_dir / best_filename
        torch.save(state, best_path)
        logger.info(f"Best model saved: {best_path}")

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    加载训练检查点（支持断点续训）。
    Load training checkpoint (supports resume training).

    Args:
        checkpoint_path (str): 检查点文件路径。Checkpoint file path.
        model (nn.Module): 要加载权重的模型。Model to load weights into.
        optimizer (Optimizer, optional): 要恢复状态的优化器（断点续训时传入）。
                                         Optimizer to restore state (pass for resume).
        scheduler (optional): 要恢复状态的学习率调度器。LR scheduler to restore.
        device (torch.device, optional): 加载到的设备，默认自动检测。
                                          Target device, auto-detect by default.

    Returns:
        Dict[str, Any]: 检查点中的完整状态字典。Full state dict from checkpoint.

    Raises:
        FileNotFoundError: 当检查点文件不存在时。When checkpoint file not found.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 确定加载设备（避免 GPU 检查点在 CPU 环境报错）
    # Determine load device (avoid error when GPU checkpoint loaded on CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # 加载优化器状态（断点续训时需要）
    # Load optimizer state (needed for resume training)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    # 加载学习率调度器状态
    # Load LR scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state loaded")

    return checkpoint


class CheckpointManager:
    """
    检查点管理器，自动管理检查点的保存和清理。
    Checkpoint manager for automatic checkpoint saving and cleanup.

    Features:
        - 定期保存检查点（每 save_freq 个 epoch）
        - 保存最佳模型（基于验证准确率）
        - 自动清理旧检查点（保留最近 keep_last 个）

    Example:
        >>> manager = CheckpointManager(save_dir="./checkpoints", keep_last=3)
        >>> for epoch in range(100):
        ...     # ... training ...
        ...     manager.save(state, epoch, val_acc)
    """

    def __init__(
        self,
        save_dir: str,
        save_freq: int = 5,
        keep_last: int = 3,
    ) -> None:
        """
        Args:
            save_dir (str): 检查点保存目录。Checkpoint save directory.
            save_freq (int): 每隔多少 epoch 保存一次。Save every N epochs.
            keep_last (int): 保留最近几个检查点（0 表示全部保留）。
                             Keep last N checkpoints (0 means keep all).
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.keep_last = keep_last
        self.best_acc = 0.0
        self._saved_checkpoints = []  # 已保存检查点路径列表（按时间顺序）

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        val_acc: float,
    ) -> None:
        """
        根据策略决定是否保存检查点。
        Decide whether to save checkpoint based on strategy.

        Args:
            state (dict): 要保存的状态字典。State dictionary to save.
            epoch (int): 当前 epoch（从 1 开始）。Current epoch (1-based).
            val_acc (float): 当前验证准确率。Current validation accuracy.
        """
        # 判断是否为最佳模型
        # Check if this is the best model
        is_best = val_acc > self.best_acc
        if is_best:
            self.best_acc = val_acc

        # 按频率保存检查点
        # Save checkpoint at specified frequency
        if epoch % self.save_freq == 0:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
            saved_path = save_checkpoint(
                state=state,
                save_dir=str(self.save_dir),
                filename=filename,
                is_best=is_best,
            )
            self._saved_checkpoints.append(saved_path)

            # 清理旧检查点（保留最近 keep_last 个）
            # Clean up old checkpoints (keep last N)
            self._cleanup_old_checkpoints()

        elif is_best:
            # 即使不在保存频率上，最佳模型也要保存
            # Save best model even if not at save frequency
            save_checkpoint(
                state=state,
                save_dir=str(self.save_dir),
                filename=f"checkpoint_epoch_{epoch:04d}.pth",
                is_best=True,
            )

    def _cleanup_old_checkpoints(self) -> None:
        """
        删除超出保留数量的旧检查点。
        Delete old checkpoints exceeding the keep limit.
        """
        if self.keep_last <= 0:
            return  # 0 表示保留所有检查点

        # 只清理常规检查点，不删除 best_model.pth
        # Only clean regular checkpoints, not best_model.pth
        while len(self._saved_checkpoints) > self.keep_last:
            old_checkpoint = self._saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")