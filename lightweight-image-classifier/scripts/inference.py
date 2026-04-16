#!/usr/bin/env python3
"""
推理脚本 / Inference Script

对单张图片或批量图片进行分类推理，输出 Top-K 预测结果。
Perform classification inference on single or batch images, output Top-K predictions.

Usage:
    # 单张图片推理 / Single image inference
    python scripts/inference.py --model resnet18 \
        --checkpoint ./checkpoints/best_model.pth \
        --image ./test.jpg --num_classes 10 --topk 5

    # 批量推理 / Batch inference
    python scripts/inference.py --model resnet18 \
        --checkpoint ./checkpoints/best_model.pth \
        --image_dir ./test_images --num_classes 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image

from datasets.transforms import get_inference_transforms
from models import build_model, get_available_models
from utils import load_checkpoint, setup_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Lightweight Image Classifier - Inference Script",
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
        "--num_classes",
        type=int,
        required=True,
        help="分类类别数量 / Number of classes",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="单张图片路径 / Single image path",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="批量推理的图片目录 / Image directory for batch inference",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="类别名称文件路径（JSON 格式，如 ['cat','dog']）/ Class names JSON file path",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="输出 Top-K 预测结果 / Output Top-K predictions",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="输入图像尺寸 / Input image size",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU 编号（-1 表示 CPU）/ GPU index (-1 for CPU)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出 JSON 文件路径 / Output JSON file path for results",
    )
    return parser.parse_args()


class ImageClassifier:
    """
    图像分类推理器，封装模型加载和推理逻辑。
    Image classifier for inference, encapsulating model loading and inference logic.

    Example:
        >>> classifier = ImageClassifier(
        ...     model_name="resnet18",
        ...     checkpoint_path="./checkpoints/best_model.pth",
        ...     num_classes=10,
        ... )
        >>> results = classifier.predict("./test.jpg", topk=5)
        >>> print(results)
    """

    # 支持的图像格式
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        num_classes: int,
        input_size: int = 224,
        device: torch.device = None,
        class_names: List[str] = None,
    ) -> None:
        """
        初始化推理器。
        Initialize classifier.

        Args:
            model_name (str): 模型名称。Model name.
            checkpoint_path (str): 检查点路径。Checkpoint path.
            num_classes (int): 类别数量。Number of classes.
            input_size (int): 输入图像尺寸。Input image size.
            device: 推理设备。Inference device.
            class_names (list): 类别名称列表。Class names list.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size

        # 加载类别名称（若未提供则使用数字索引）
        # Load class names (use numeric indices if not provided)
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = [str(i) for i in range(num_classes)]

        # 构建并加载模型
        # Build and load model
        self.model = build_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
        self.model = self.model.to(device)

        # 从检查点加载权重
        # Load weights from checkpoint
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            device=device,
        )

        # 尝试从检查点中读取类别名称
        # Try to read class names from checkpoint
        if "class_names" in checkpoint and class_names is None:
            self.class_names = checkpoint["class_names"]

        # 切换到评估模式
        # Switch to evaluation mode
        self.model.eval()

        # 构建推理预处理变换
        # Build inference preprocessing transforms
        self.transform = get_inference_transforms(input_size=input_size)

    def predict(
        self,
        image_path: str,
        topk: int = 5,
    ) -> List[Dict]:
        """
        对单张图片进行分类推理。
        Perform classification inference on a single image.

        Args:
            image_path (str): 图片路径。Image path.
            topk (int): 返回 Top-K 预测结果。Return Top-K predictions.

        Returns:
            List[Dict]: Top-K 预测结果列表，每项包含 rank、class_name、class_idx、confidence。
                        Top-K prediction list, each item contains rank, class_name, class_idx, confidence.
        """
        # 加载并预处理图像
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)  # 添加 batch 维度
        input_tensor = input_tensor.to(self.device)

        # 推理（禁用梯度计算）
        # Inference (disable gradient computation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # 将 logits 转换为概率
            # Convert logits to probabilities
            probabilities = F.softmax(outputs, dim=1)

        # 获取 Top-K 结果
        # Get Top-K results
        k = min(topk, self.num_classes)
        top_probs, top_indices = probabilities.topk(k, dim=1)

        top_probs = top_probs.squeeze(0).cpu().tolist()
        top_indices = top_indices.squeeze(0).cpu().tolist()

        results = []
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
            results.append({
                "rank": rank,
                "class_idx": idx,
                "class_name": self.class_names[idx],
                "confidence": round(prob * 100, 2),  # 转换为百分比
            })

        return results

    def predict_batch(
        self,
        image_dir: str,
        topk: int = 5,
    ) -> List[Dict]:
        """
        对目录中的所有图片进行批量推理。
        Perform batch inference on all images in a directory.

        Args:
            image_dir (str): 图片目录路径。Image directory path.
            topk (int): 每张图片返回 Top-K 预测结果。Top-K predictions per image.

        Returns:
            List[Dict]: 每张图片的推理结果列表。Inference results for each image.
        """
        image_dir = Path(image_dir)
        image_files = [
            f for f in sorted(image_dir.iterdir())
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            raise RuntimeError(f"No supported images found in: {image_dir}")

        all_results = []
        for img_path in image_files:
            start_time = time.time()
            try:
                preds = self.predict(str(img_path), topk=topk)
                elapsed_ms = (time.time() - start_time) * 1000
                all_results.append({
                    "image": img_path.name,
                    "predictions": preds,
                    "inference_time_ms": round(elapsed_ms, 2),
                })
            except Exception as e:
                all_results.append({
                    "image": img_path.name,
                    "error": str(e),
                })

        return all_results


def main() -> None:
    """主推理函数。Main inference function."""
    args = parse_args()

    logger = setup_logger(name="image_classifier", log_to_file=False)
    logger.info("PyTorch Lightweight Image Classifier - Inference")

    # 设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 加载类别名称（可选）
    # Load class names (optional)
    class_names = None
    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = json.load(f)

    # 初始化推理器
    # Initialize classifier
    classifier = ImageClassifier(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        input_size=args.input_size,
        device=device,
        class_names=class_names,
    )

    results = []

    if args.image:
        # 单张图片推理
        # Single image inference
        logger.info(f"Inferring: {args.image}")
        start = time.time()
        preds = classifier.predict(args.image, topk=args.topk)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(f"\nResults for: {args.image}")
        logger.info(f"Inference time: {elapsed_ms:.1f} ms")
        logger.info("-" * 40)
        for pred in preds:
            logger.info(
                f"  Top-{pred['rank']}: {pred['class_name']:<20} "
                f"({pred['confidence']:.2f}%)"
            )

        results = [{"image": args.image, "predictions": preds}]

    elif args.image_dir:
        # 批量推理
        # Batch inference
        logger.info(f"Batch inferring from: {args.image_dir}")
        results = classifier.predict_batch(args.image_dir, topk=args.topk)

        for item in results:
            if "error" in item:
                logger.warning(f"  {item['image']}: ERROR - {item['error']}")
            else:
                top1 = item["predictions"][0]
                logger.info(
                    f"  {item['image']:<40} -> "
                    f"{top1['class_name']} ({top1['confidence']:.2f}%) "
                    f"[{item['inference_time_ms']:.1f}ms]"
                )
    else:
        logger.error("Please specify --image or --image_dir")
        sys.exit(1)

    # 保存结果到 JSON
    # Save results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()