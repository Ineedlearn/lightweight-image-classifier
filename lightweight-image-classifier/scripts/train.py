#!/usr/bin/env python3
"""
训练脚本 / Training Script

完整的图像分类模型训练流程，支持：
- 多模型一键切换（ResNet/MobileNet/ShuffleNet）
- 断点续训（--resume 参数）
- 混合精度训练（--amp 参数）
- 学习率调度（step/cosine/plateau）
- TensorBoard 日志记录
- 自动保存最佳模型和定期检查点

Full image classification training pipeline supporting:
- Multi-model one-click switching (ResNet/MobileNet/ShuffleNet)
- Resume training (--resume argument)
- Mixed precision training (--amp argument)
- LR scheduling (step/cosine/plateau)
- TensorBoard logging
- Auto-save best model and periodic checkpoints

Usage:
    python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10
    python scripts/train.py --model mobilenet_v2 --data_dir ./data --num_classes 10 --amp
    python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10 \
        --resume ./checkpoints/checkpoint_epoch_0010.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 将项目根目录加入 Python 路径，确保模块可以被正确导入
# Add project root to Python path for correct module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from datasets import create_dataloaders
from models import build_model, get_available_models
from utils import (
    AverageMeter,
    CheckpointManager,
    TrainingVisualizer,
    accuracy,
    load_checkpoint,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    Parse command line arguments.

    Returns:
        argparse.Namespace: 解析后的参数对象。Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Lightweight Image Classifier - Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模型参数 / Model Arguments ----
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=get_available_models(),
        help="模型名称 / Model name",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="使用 ImageNet 预训练权重 / Use ImageNet pretrained weights",
    )
    parser.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
        help="不使用预训练权重 / Do not use pretrained weights",
    )

    # ---- 数据参数 / Data Arguments ----
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据集根目录（需含 train/ 和 val/ 子目录）/ Dataset root directory",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="分类类别数量 / Number of classification classes",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="输入图像尺寸 / Input image size",
    )

    # ---- 训练参数 / Training Arguments ----
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="总训练轮数 / Total training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批大小 / Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="初始学习率 / Initial learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD 动量 / SGD momentum",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减 / Weight decay",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="开启自动混合精度训练 / Enable automatic mixed precision training",
    )

    # ---- 学习率调度参数 / LR Scheduler Arguments ----
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine", "plateau"],
        help="学习率调度策略 / LR scheduler type",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="StepLR 衰减步长 / StepLR decay step size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="StepLR 衰减系数 / StepLR decay gamma",
    )

    # ---- 断点续训参数 / Resume Arguments ----
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练的文件路径 / Checkpoint path to resume training from",
    )

    # ---- 保存参数 / Save Arguments ----
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="检查点保存目录 / Checkpoint save directory",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5,
        help="检查点保存频率（每 N 个 epoch）/ Checkpoint save frequency (every N epochs)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="日志保存目录 / Log save directory",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="./runs",
        help="TensorBoard 日志目录 / TensorBoard log directory",
    )

    # ---- 系统参数 / System Arguments ----
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 工作进程数 / DataLoader worker processes",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的 GPU 编号（-1 表示使用 CPU）/ GPU index (-1 for CPU)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 / Random seed",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保实验可复现。
    Set random seed for reproducibility.

    Args:
        seed (int): 随机种子值。Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 注意：以下设置会降低训练速度，但确保完全可复现
        # Note: Following settings reduce training speed but ensure full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int) -> torch.device:
    """
    获取训练设备。
    Get training device.

    Args:
        gpu_id (int): GPU 编号，-1 表示使用 CPU。GPU index, -1 for CPU.

    Returns:
        torch.device: 训练设备。Training device.
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    return device


def build_optimizer(
    model: nn.Module,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    构建 SGD 优化器（含动量和权重衰减）。
    Build SGD optimizer with momentum and weight decay.

    Args:
        model (nn.Module): 模型。Model.
        lr (float): 初始学习率。Initial learning rate.
        momentum (float): 动量系数。Momentum coefficient.
        weight_decay (float): 权重衰减系数。Weight decay coefficient.

    Returns:
        optim.Optimizer: SGD 优化器。SGD optimizer.
    """
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,  # 使用 Nesterov 动量（通常比标准动量收敛更快）
    )


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    step_size: int,
    gamma: float,
) -> optim.lr_scheduler._LRScheduler:
    """
    构建学习率调度器。
    Build learning rate scheduler.

    Args:
        optimizer: 优化器。Optimizer.
        scheduler_type (str): 调度类型（step/cosine/plateau）。Scheduler type.
        epochs (int): 总训练轮数。Total training epochs.
        step_size (int): StepLR 步长。StepLR step size.
        gamma (float): 衰减系数。Decay factor.

    Returns:
        LR scheduler.
    """
    if scheduler_type == "step":
        # 每 step_size 个 epoch 将 lr 乘以 gamma
        # Multiply lr by gamma every step_size epochs
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "cosine":
        # 余弦退火：lr 从初始值余弦衰减到 eta_min
        # Cosine annealing: lr decays from initial to eta_min following cosine curve
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "plateau":
        # 当验证指标不再改善时降低 lr
        # Reduce lr when validation metric stops improving
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: GradScaler,
    use_amp: bool,
    log_freq: int = 10,
    logger=None,
) -> tuple:
    """
    执行一个 epoch 的训练。
    Execute one epoch of training.

    Args:
        model: 模型。Model.
        loader: 训练 DataLoader。Training DataLoader.
        criterion: 损失函数。Loss function.
        optimizer: 优化器。Optimizer.
        device: 训练设备。Training device.
        epoch (int): 当前 epoch（从 1 开始）。Current epoch (1-based).
        total_epochs (int): 总 epoch 数。Total epochs.
        scaler: AMP GradScaler。AMP GradScaler.
        use_amp (bool): 是否使用混合精度。Whether to use mixed precision.
        log_freq (int): 打印日志的频率（每 N 个 batch）。Log frequency.
        logger: 日志记录器。Logger.

    Returns:
        tuple: (平均损失, Top-1 准确率) / (average loss, Top-1 accuracy)
    """
    model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 等）

    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc@1")

    total_batches = len(loader)
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        # 将数据移动到目标设备
        # Move data to target device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 前向传播（支持混合精度）
        # Forward pass (with mixed precision support)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 反向传播和参数更新
        # Backward pass and parameter update
        optimizer.zero_grad()

        if use_amp:
            # AMP 模式：使用 GradScaler 缩放梯度，防止下溢
            # AMP mode: use GradScaler to scale gradients, prevent underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 计算准确率
        # Compute accuracy
        batch_size = images.size(0)
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]

        # 更新统计量
        # Update statistics
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc1.item(), batch_size)

        # 定期打印训练日志
        # Periodically print training log
        if (batch_idx + 1) % log_freq == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
            log_msg = (
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{total_batches}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"Acc@1: {acc_meter.avg:.2f}% "
                f"ETA: {eta:.0f}s"
            )
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    logger=None,
) -> tuple:
    """
    在验证集上评估模型。
    Evaluate model on validation set.

    使用 @torch.no_grad() 装饰器禁用梯度计算，节省内存和加速推理。
    Use @torch.no_grad() decorator to disable gradient computation.

    Args:
        model: 模型。Model.
        loader: 验证 DataLoader。Validation DataLoader.
        criterion: 损失函数。Loss function.
        device: 设备。Device.
        logger: 日志记录器。Logger.

    Returns:
        tuple: (平均损失, Top-1 准确率, Top-5 准确率) / (avg loss, top1 acc, top5 acc)
    """
    model.eval()  # 切换到评估模式（禁用 Dropout、固定 BatchNorm）

    loss_meter = AverageMeter("Val Loss")
    top1_meter = AverageMeter("Val Acc@1")
    top5_meter = AverageMeter("Val Acc@5")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 前向传播
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 计算 Top-1 和 Top-5 准确率
        # Compute Top-1 and Top-5 accuracy
        batch_size = images.size(0)
        # 当类别数 < 5 时，只计算 Top-1
        # When num_classes < 5, only compute Top-1
        k = min(5, outputs.size(1))
        topk_accs = accuracy(outputs, targets, topk=(1, k))
        acc1 = topk_accs[0]
        acc5 = topk_accs[-1]

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

    log_msg = (
        f"Validation -> Loss: {loss_meter.avg:.4f} "
        f"Acc@1: {top1_meter.avg:.2f}% "
        f"Acc@5: {top5_meter.avg:.2f}%"
    )
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def main() -> None:
    """主训练函数。Main training function."""
    # ---- 1. 解析参数 / Parse Arguments ----
    args = parse_args()

    # ---- 2. 设置随机种子 / Set Random Seed ----
    set_seed(args.seed)

    # ---- 3. 配置日志 / Setup Logger ----
    logger = setup_logger(
        name="image_classifier",
        log_dir=args.log_dir,
        log_to_file=True,
    )
    logger.info("=" * 60)
    logger.info("PyTorch Lightweight Image Classifier - Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    # ---- 4. 获取训练设备 / Get Training Device ----
    device = get_device(args.gpu)

    # ---- 5. 创建数据加载器 / Create DataLoaders ----
    logger.info(f"Loading dataset from: {args.data_dir}")
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(f"Classes ({args.num_classes}): {class_names}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- 6. 构建模型 / Build Model ----
    logger.info(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    # 统计参数量
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ---- 7. 定义损失函数和优化器 / Define Loss and Optimizer ----
    # 交叉熵损失（适用于多分类问题）
    # Cross-entropy loss (for multi-class classification)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    # AMP GradScaler（混合精度训练时使用）
    # AMP GradScaler (used for mixed precision training)
    scaler = GradScaler(enabled=args.amp)

    # ---- 8. 断点续训 / Resume Training ----
    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        # 从检查点恢复训练状态
        # Restore training state from checkpoint
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best_acc: {best_acc:.2f}%")

    # ---- 9. 初始化辅助工具 / Initialize Auxiliary Tools ----
    # TensorBoard 写入器
    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    # 检查点管理器
    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        keep_last=3,
    )

    # 训练可视化器
    # Training visualizer
    visualizer = TrainingVisualizer(save_dir=args.log_dir)

    # ---- 10. 主训练循环 / Main Training Loop ----
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    logger.info(f"Mixed Precision (AMP): {args.amp}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 获取当前学习率
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch [{epoch}/{args.epochs}] | LR: {current_lr:.6f}")

        # 训练一个 epoch
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
            use_amp=args.amp,
            logger=logger,
        )

        # 在验证集上评估
        # Evaluate on validation set
        val_loss, val_acc, val_acc5 = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
        )

        # 更新学习率调度器
        # Update LR scheduler
        if args.scheduler == "plateau":
            # ReduceLROnPlateau 需要传入监控指标
            # ReduceLROnPlateau requires the monitored metric
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # 记录到 TensorBoard
        # Log to TensorBoard
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
        tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
        tb_writer.add_scalar("Accuracy/val_top5", val_acc5, epoch)
        tb_writer.add_scalar("LR", current_lr, epoch)

        # 更新可视化器
        # Update visualizer
        visualizer.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=current_lr,
        )

        # 判断是否为最佳模型
        # Check if this is the best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            logger.info(f"✓ New best model! Val Acc@1: {best_acc:.2f}%")

        # 保存检查点
        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_name": args.model,
            "num_classes": args.num_classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "val_acc": val_acc,
            "class_names": class_names,
        }
        ckpt_manager.save(
            state=checkpoint_state,
            epoch=epoch,
            val_acc=val_acc,
        )

        # 打印 epoch 摘要
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Best Acc: {best_acc:.2f}% | Time: {epoch_time:.1f}s"
        )

        # 每 10 个 epoch 保存一次训练曲线图
        # Save training curves every 10 epochs
        if epoch % 10 == 0:
            visualizer.plot_all()

    # ---- 11. 训练结束 / Training Complete ----
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")
    logger.info(f"Best model saved to: {args.save_dir}/best_model.pth")
    logger.info("=" * 60)

    # 保存最终训练曲线
    # Save final training curves
    visualizer.plot_all()

    # 关闭 TensorBoard 写入器
    # Close TensorBoard writer
    tb_writer.close()

    # 打印 TensorBoard 启动提示
    # Print TensorBoard launch hint
    logger.info(f"\nTo view training curves in TensorBoard, run:")
    logger.info(f"  tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()