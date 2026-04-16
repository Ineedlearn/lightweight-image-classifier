#!/usr/bin/env python3
"""
ONNX 导出脚本 / ONNX Export Script

将训练好的 PyTorch 模型导出为 ONNX 格式，便于跨平台部署。
Export trained PyTorch model to ONNX format for cross-platform deployment.

Usage:
    python scripts/export_onnx.py --model resnet18 \
        --checkpoint ./checkpoints/best_model.pth \
        --num_classes 10 --output model.onnx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models import build_model, get_available_models
from utils import load_checkpoint, setup_logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数。Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch Lightweight Image Classifier - ONNX Export Script",
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
        "--output",
        type=str,
        default="model.onnx",
        help="ONNX 输出文件路径 / ONNX output file path",
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
        default=1,
        help="导出时的批大小（1 表示单张推理）/ Export batch size",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        default=False,
        help="是否支持动态批大小 / Whether to support dynamic batch size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本 / ONNX opset version",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="导出后验证 ONNX 模型 / Verify ONNX model after export",
    )
    return parser.parse_args()


def export_to_onnx(
    model,
    output_path: str,
    input_size: int = 224,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    opset: int = 11,
    logger=None,
) -> None:
    """
    将 PyTorch 模型导出为 ONNX 格式。
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch 模型（已切换到 eval 模式）。PyTorch model (in eval mode).
        output_path (str): ONNX 文件输出路径。ONNX file output path.
        input_size (int): 输入图像尺寸。Input image size.
        batch_size (int): 批大小。Batch size.
        dynamic_batch (bool): 是否使用动态批大小。Whether to use dynamic batch size.
        opset (int): ONNX opset 版本。ONNX opset version.
        logger: 日志记录器。Logger.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建虚拟输入张量（用于追踪计算图）
    # Create dummy input tensor (for tracing computation graph)
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # 定义动态轴（若启用动态批大小）
    # Define dynamic axes (if dynamic batch size enabled)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    if logger:
        logger.info(f"Exporting model to ONNX...")
        logger.info(f"  Input shape: {list(dummy_input.shape)}")
        logger.info(f"  ONNX opset: {opset}")
        logger.info(f"  Dynamic batch: {dynamic_batch}")

    # 执行 ONNX 导出
    # Perform ONNX export
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(output_path),
        export_params=True,          # 导出模型权重
        opset_version=opset,         # ONNX opset 版本
        do_constant_folding=True,    # 常量折叠优化
        input_names=["input"],       # 输入节点名称
        output_names=["output"],     # 输出节点名称
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    if logger:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model exported: {output_path} ({file_size_mb:.2f} MB)")


def verify_onnx(output_path: str, input_size: int, batch_size: int, logger=None) -> bool:
    """
    验证导出的 ONNX 模型是否有效。
    Verify that the exported ONNX model is valid.

    Args:
        output_path (str): ONNX 文件路径。ONNX file path.
        input_size (int): 输入图像尺寸。Input image size.
        batch_size (int): 批大小。Batch size.
        logger: 日志记录器。Logger.

    Returns:
        bool: 验证是否通过。Whether verification passed.
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        # 1. 使用 onnx 库验证模型结构
        # 1. Validate model structure using onnx library
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        if logger:
            logger.info("ONNX model structure check: PASSED")

        # 2. 使用 onnxruntime 进行推理验证
        # 2. Validate inference using onnxruntime
        ort_session = ort.InferenceSession(output_path)
        dummy_input = np.random.randn(
            batch_size, 3, input_size, input_size
        ).astype(np.float32)

        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_outputs = ort_session.run(None, ort_inputs)

        if logger:
            logger.info(f"ONNX runtime inference check: PASSED")
            logger.info(f"  Output shape: {ort_outputs[0].shape}")

        return True

    except ImportError:
        if logger:
            logger.warning("onnx or onnxruntime not installed, skipping verification")
            logger.warning("Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        if logger:
            logger.error(f"ONNX verification failed: {e}")
        return False


def main() -> None:
    """主导出函数。Main export function."""
    args = parse_args()

    logger = setup_logger(name="image_classifier", log_to_file=False)
    logger.info("PyTorch Lightweight Image Classifier - ONNX Export")
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device("cpu")  # ONNX 导出通常在 CPU 上进行

    # 构建模型
    # Build model
    logger.info(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=False,
    )
    model = model.to(device)

    # 加载检查点
    # Load checkpoint
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )

    # 切换到评估模式（禁用 Dropout 和 BatchNorm 的训练行为）
    # Switch to eval mode (disable Dropout and BatchNorm training behavior)
    model.eval()

    # 导出 ONNX
    # Export ONNX
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_size=args.input_size,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        opset=args.opset,
        logger=logger,
    )

    # 验证导出结果
    # Verify export result
    if args.verify:
        logger.info("\nVerifying ONNX model...")
        success = verify_onnx(
            output_path=args.output,
            input_size=args.input_size,
            batch_size=args.batch_size,
            logger=logger,
        )
        if success:
            logger.info("Export and verification completed successfully!")
        else:
            logger.warning("Export completed but verification failed or skipped.")

    logger.info(f"\nONNX model saved to: {args.output}")
    logger.info("You can now use this model with ONNX Runtime for inference.")


if __name__ == "__main__":
    main()