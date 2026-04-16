#!/usr/bin/env python3
"""
验证脚本 / Validation Script

在验证集上评估训练好的模型，输出准确率、损失和混淆矩阵。
Evaluate trained model on validation set, output accuracy, loss, and confusion matrix.

Usage:
    python scripts/validate.py --model resnet18 \
        --checkpoint ./checkpoints/best_model.pth \
        --data_dir ./data --num_classes 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from datasets import create_dataloaders
from models import build_model, get_available_models
from utils import (
    AverageMeter,
    TrainingVisualizer,
    accuracy,
    compute_confusion_matrix,
    load_checkpoint,
    setup_logger,
)
from utils.metrics import compute_per_class_accuracy


def parse_args() -> argparse.Namespace:
    """解析命令行参数。Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Lightweight Image Classifier - Validation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_available_models(),
        help="模型名称 / Model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径 / Model checkpoint path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据集根目录 / Dataset root directory",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="分类类别数量 / Number of classes",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="输入图像尺寸 / Input image size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批大小 / Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 工作进程数 / DataLoader workers",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU 编号（-1 表示 CPU）/ GPU index (-1 for CPU)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs",
        help="结果保存目录 / Results save directory",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list,
    logger,
) -> dict:
    """
    完整评估模型，返回详细指标。
    Fully evaluate model, return detailed metrics.

    Args:
        model: 模型。Model.
        loader: 验证 DataLoader。Validation DataLoader.
        criterion: 损失函数。Loss function.
        device: 设备。Device.
        num_classes (int): 类别数。Number of classes.
        class_names (list): 类别名称列表。Class names.
        logger: 日志记录器。Logger.

    Returns:
        dict: 包含各项评估指标的字典。Dict containing evaluation metrics.
    """
    model.eval()

    loss_meter = AverageMeter("Loss")
    top1_meter = AverageMeter("Acc@1")
    top5_meter = AverageMeter("Acc@5")

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        k = min(5, outputs.size(1))
        topk_accs = accuracy(outputs, targets, topk=(1, k))

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(topk_accs[0].item(), batch_size)
        top5_meter.update(topk_accs[-1].item(), batch_size)

        # 收集预测结果用于混淆矩阵
        # Collect predictions for confusion matrix
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().tolist())

    # 计算混淆矩阵和每类准确率
    # Compute confusion matrix and per-class accuracy
    cm = compute_confusion_matrix(all_preds, all_targets, num_classes)
    per_class_acc = compute_per_class_accuracy(cm)

    # 打印结果
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Validation Results")
    logger.info("=" * 60)
    logger.info(f"  Loss:   {loss_meter.avg:.4f}")
    logger.info(f"  Acc@1:  {top1_meter.avg:.2f}%")
    logger.info(f"  Acc@5:  {top5_meter.avg:.2f}%")
    logger.info("\nPer-Class Accuracy:")
    for cls_name, cls_acc in zip(class_names, per_class_acc):
        logger.info(f"  {cls_name:<30} {cls_acc * 100:.2f}%")
    logger.info("=" * 60)

    return {
        "loss": loss_meter.avg,
        "acc1": top1_meter.avg,
        "acc5": top5_meter.avg,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
    }


def main() -> None:
    """主验证函数。Main validation function."""
    args = parse_args()

    logger = setup_logger(
        name="image_classifier",
        log_dir=args.save_dir,
        log_to_file=True,
    )
    logger.info("PyTorch Lightweight Image Classifier - Validation")
    logger.info(f"Arguments: {vars(args)}")

    # 设备
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # 数据加载器（只需要验证集）
    # DataLoader (only need validation set)
    _, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 构建模型并加载权重
    # Build model and load weights
    logger.info(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=False,  # 验证时不需要预训练权重，直接加载 checkpoint
    )
    model = model.to(device)

    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    # 执行评估
    # Run evaluation
    results = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=args.num_classes,
        class_names=class_names,
        logger=logger,
    )

    # 绘制混淆矩阵
    # Plot confusion matrix
    visualizer = TrainingVisualizer(save_dir=args.save_dir)
    visualizer.plot_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        filename="val_confusion_matrix.png",
    )
    logger.info(f"Confusion matrix saved to: {args.save_dir}/val_confusion_matrix.png")


if __name__ == "__main__":
    main()