"""
评估指标模块 / Evaluation Metrics Module

提供训练过程中常用的评估指标计算工具。
Provides common evaluation metric calculation utilities for training.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AverageMeter:
    """
    跟踪并计算指标的滑动平均值。
    Track and compute running average of metrics.

    常用于记录每个 epoch 的平均 loss 和 accuracy。
    Commonly used to record average loss and accuracy per epoch.

    Example:
        >>> loss_meter = AverageMeter("Loss")
        >>> for batch in dataloader:
        ...     loss = criterion(output, target)
        ...     loss_meter.update(loss.item(), n=batch_size)
        >>> print(f"Avg Loss: {loss_meter.avg:.4f}")
    """

    def __init__(self, name: str = "", fmt: str = ":f") -> None:
        """
        Args:
            name (str): 指标名称（用于打印）。Metric name (for printing).
            fmt (str): 格式化字符串。Format string.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """重置所有统计量。Reset all statistics."""
        self.val = 0.0      # 当前批次值
        self.avg = 0.0      # 累计平均值
        self.sum = 0.0      # 累计总和
        self.count = 0      # 累计样本数

    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计量。
        Update statistics.

        Args:
            val (float): 当前批次的指标值（通常是均值）。Current batch metric value (usually mean).
            n (int): 当前批次的样本数量。Number of samples in current batch.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[torch.Tensor]:
    """
    计算 Top-K 准确率。
    Compute Top-K accuracy.

    Args:
        output (torch.Tensor): 模型输出 logits，形状 (N, C)。
                               Model output logits, shape (N, C).
        target (torch.Tensor): 真实标签，形状 (N,)。
                               Ground truth labels, shape (N,).
        topk (tuple): 要计算的 K 值列表，如 (1, 5)。
                      K values to compute, e.g., (1, 5).

    Returns:
        List[torch.Tensor]: 各 K 值对应的准确率（百分比）。
                            Accuracy (percentage) for each K value.

    Example:
        >>> output = torch.randn(32, 10)  # batch_size=32, num_classes=10
        >>> target = torch.randint(0, 10, (32,))
        >>> top1, top5 = accuracy(output, target, topk=(1, 5))
        >>> print(f"Top-1: {top1.item():.2f}%, Top-5: {top5.item():.2f}%")
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 获取 Top-K 预测结果（按置信度降序排列）
        # Get Top-K predictions (sorted by confidence in descending order)
        # output: (N, C) -> topk_pred: (N, maxk)
        _, topk_pred = output.topk(maxk, dim=1, largest=True, sorted=True)

        # 转置以便比较：(N, maxk) -> (maxk, N)
        # Transpose for comparison: (N, maxk) -> (maxk, N)
        topk_pred = topk_pred.t()

        # 与真实标签比较
        # Compare with ground truth labels
        # target.view(1, -1): (1, N) -> expand to (maxk, N)
        correct = topk_pred.eq(target.view(1, -1).expand_as(topk_pred))

        results = []
        for k in topk:
            # 取前 k 行中任意一行正确即算正确
            # A sample is correct if any of the top-k predictions matches
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 转换为百分比
            # Convert to percentage
            results.append(correct_k.mul_(100.0 / batch_size))

        return results


def compute_confusion_matrix(
    predictions: List[int],
    targets: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    计算混淆矩阵。
    Compute confusion matrix.

    Args:
        predictions (List[int]): 预测标签列表。Predicted label list.
        targets (List[int]): 真实标签列表。Ground truth label list.
        num_classes (int): 类别数量。Number of classes.

    Returns:
        np.ndarray: 混淆矩阵，形状 (num_classes, num_classes)。
                    Confusion matrix, shape (num_classes, num_classes).
                    matrix[i][j] 表示真实类别为 i 但预测为 j 的样本数。
                    matrix[i][j] = number of samples with true class i predicted as j.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(predictions, targets):
        confusion[true][pred] += 1
    return confusion


def compute_per_class_accuracy(
    confusion_matrix: np.ndarray,
) -> np.ndarray:
    """
    从混淆矩阵计算每个类别的准确率。
    Compute per-class accuracy from confusion matrix.

    Args:
        confusion_matrix (np.ndarray): 混淆矩阵。Confusion matrix.

    Returns:
        np.ndarray: 每个类别的准确率。Per-class accuracy.
    """
    # 对角线元素为正确预测数，行和为该类别总样本数
    # Diagonal elements are correct predictions, row sum is total samples per class
    per_class_correct = np.diag(confusion_matrix)
    per_class_total = confusion_matrix.sum(axis=1)

    # 避免除以零（某类别无样本时）
    # Avoid division by zero (when a class has no samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(
            per_class_total > 0,
            per_class_correct / per_class_total,
            0.0,
        )

    return per_class_acc