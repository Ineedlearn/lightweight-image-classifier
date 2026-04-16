"""
训练可视化模块 / Training Visualization Module

提供训练过程中的指标可视化，包括损失曲线、准确率曲线和混淆矩阵。
Provides training process visualization including loss curves, accuracy curves, and confusion matrix.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 使用非交互式后端（适合服务器环境）
# Use non-interactive backend (suitable for server environments)
matplotlib.use("Agg")

logger = logging.getLogger("image_classifier")


class TrainingVisualizer:
    """
    训练过程可视化器，自动记录并绘制训练曲线。
    Training process visualizer for automatic recording and plotting of training curves.

    支持的可视化 / Supported visualizations:
        - 训练/验证损失曲线
        - 训练/验证准确率曲线
        - 学习率变化曲线
        - 混淆矩阵热力图

    Example:
        >>> visualizer = TrainingVisualizer(save_dir="./logs")
        >>> for epoch in range(100):
        ...     visualizer.update(epoch, train_loss, val_loss, train_acc, val_acc, lr)
        >>> visualizer.plot_all()
    """

    def __init__(self, save_dir: str = "./logs") -> None:
        """
        Args:
            save_dir (str): 图表保存目录。Chart save directory.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 历史记录
        # History records
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.learning_rates: List[float] = []

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """
        更新训练历史记录。
        Update training history records.

        Args:
            epoch (int): 当前 epoch。Current epoch.
            train_loss (float): 训练损失。Training loss.
            val_loss (float): 验证损失。Validation loss.
            train_acc (float): 训练准确率（%）。Training accuracy (%).
            val_acc (float): 验证准确率（%）。Validation accuracy (%).
            lr (float): 当前学习率。Current learning rate.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

    def plot_all(self, show: bool = False) -> None:
        """
        绘制所有训练曲线并保存。
        Plot all training curves and save.

        Args:
            show (bool): 是否显示图表（服务器环境通常设为 False）。
                         Whether to display charts (usually False on servers).
        """
        self.plot_loss_curve(show=show)
        self.plot_accuracy_curve(show=show)
        self.plot_combined_curve(show=show)
        self.plot_lr_curve(show=show)
        logger.info(f"Training curves saved to: {self.save_dir}")

    def plot_loss_curve(
        self,
        filename: str = "loss_curve.png",
        show: bool = False,
    ) -> None:
        """
        绘制训练/验证损失曲线。
        Plot training/validation loss curves.

        Args:
            filename (str): 保存文件名。Save filename.
            show (bool): 是否显示。Whether to display.
        """
        if not self.epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.epochs, self.train_losses, "b-o", label="Train Loss",
                linewidth=2, markersize=4)
        ax.plot(self.epochs, self.val_losses, "r-s", label="Val Loss",
                linewidth=2, markersize=4)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 标注最低验证损失点
        # Mark the lowest validation loss point
        min_val_loss_idx = np.argmin(self.val_losses)
        ax.annotate(
            f"Min: {self.val_losses[min_val_loss_idx]:.4f}",
            xy=(self.epochs[min_val_loss_idx], self.val_losses[min_val_loss_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="red",
            arrowprops={"arrowstyle": "->", "color": "red"},
        )

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_accuracy_curve(
        self,
        filename: str = "accuracy_curve.png",
        show: bool = False,
    ) -> None:
        """
        绘制训练/验证准确率曲线。
        Plot training/validation accuracy curves.

        Args:
            filename (str): 保存文件名。Save filename.
            show (bool): 是否显示。Whether to display.
        """
        if not self.epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.epochs, self.train_accs, "b-o", label="Train Acc",
                linewidth=2, markersize=4)
        ax.plot(self.epochs, self.val_accs, "r-s", label="Val Acc",
                linewidth=2, markersize=4)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # 标注最高验证准确率点
        # Mark the highest validation accuracy point
        max_val_acc_idx = np.argmax(self.val_accs)
        ax.annotate(
            f"Best: {self.val_accs[max_val_acc_idx]:.2f}%",
            xy=(self.epochs[max_val_acc_idx], self.val_accs[max_val_acc_idx]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
            color="red",
            arrowprops={"arrowstyle": "->", "color": "red"},
        )

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_combined_curve(
        self,
        filename: str = "training_curves.png",
        show: bool = False,
    ) -> None:
        """
        绘制损失和准确率的组合图（2x1 布局）。
        Plot combined loss and accuracy charts (2x1 layout).

        Args:
            filename (str): 保存文件名。Save filename.
            show (bool): 是否显示。Whether to display.
        """
        if not self.epochs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        # 左图：损失曲线
        # Left: loss curves
        ax1.plot(self.epochs, self.train_losses, "b-o", label="Train Loss",
                 linewidth=2, markersize=3)
        ax1.plot(self.epochs, self.val_losses, "r-s", label="Val Loss",
                 linewidth=2, markersize=3)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title("Loss Curves", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 右图：准确率曲线
        # Right: accuracy curves
        ax2.plot(self.epochs, self.train_accs, "b-o", label="Train Acc",
                 linewidth=2, markersize=3)
        ax2.plot(self.epochs, self.val_accs, "r-s", label="Val Acc",
                 linewidth=2, markersize=3)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_title("Accuracy Curves", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_lr_curve(
        self,
        filename: str = "lr_curve.png",
        show: bool = False,
    ) -> None:
        """
        绘制学习率变化曲线。
        Plot learning rate change curve.

        Args:
            filename (str): 保存文件名。Save filename.
            show (bool): 是否显示。Whether to display.
        """
        if not self.epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.semilogy(self.epochs, self.learning_rates, "g-o",
                    linewidth=2, markersize=4)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate (log scale)", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        filename: str = "confusion_matrix.png",
        show: bool = False,
    ) -> None:
        """
        绘制混淆矩阵热力图。
        Plot confusion matrix heatmap.

        Args:
            confusion_matrix (np.ndarray): 混淆矩阵。Confusion matrix.
            class_names (List[str]): 类别名称列表。Class names list.
            filename (str): 保存文件名。Save filename.
            show (bool): 是否显示。Whether to display.
        """
        import seaborn as sns

        n_classes = len(class_names)
        fig_size = max(8, n_classes * 0.8)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # 归一化混淆矩阵（每行除以该行总和）
        # Normalize confusion matrix (divide each row by row sum)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_cm = np.where(
            row_sums > 0,
            confusion_matrix / row_sums,
            0.0,
        )

        sns.heatmap(
            normalized_cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        logger.info(f"Confusion matrix saved: {save_path}")