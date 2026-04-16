"""
模型工厂模块 / Model Factory Module

提供统一的模型创建接口，支持一键切换多种模型。
Provides unified model creation interface supporting one-click model switching.

支持的模型 / Supported Models:
    ResNet 系列: resnet18, resnet34, resnet50
    MobileNet 系列: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    ShuffleNet 系列: shufflenet_v2_x0_5, shufflenet_v2_x1_0
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.mobilenet import build_mobilenet
from models.resnet import build_resnet
from models.shufflenet import build_shufflenet


# 模型注册表：模型名称 -> (构建函数, 描述)
# Model registry: model name -> (build function, description)
MODEL_REGISTRY: Dict[str, dict] = {
    # ResNet 系列
    "resnet18": {
        "builder": build_resnet,
        "variant": "resnet18",
        "params": "11.7M",
        "description": "ResNet-18: 轻量级，适合快速实验",
    },
    "resnet34": {
        "builder": build_resnet,
        "variant": "resnet34",
        "params": "21.8M",
        "description": "ResNet-34: 中等规模，精度与速度均衡",
    },
    "resnet50": {
        "builder": build_resnet,
        "variant": "resnet50",
        "params": "25.6M",
        "description": "ResNet-50: 较高精度，适合生产部署",
    },
    # MobileNet 系列
    "mobilenet_v2": {
        "builder": build_mobilenet,
        "variant": "mobilenet_v2",
        "params": "3.4M",
        "description": "MobileNetV2: 极轻量，适合移动端部署",
    },
    "mobilenet_v3_small": {
        "builder": build_mobilenet,
        "variant": "mobilenet_v3_small",
        "params": "2.5M",
        "description": "MobileNetV3-Small: 最轻量，适合边缘设备",
    },
    "mobilenet_v3_large": {
        "builder": build_mobilenet,
        "variant": "mobilenet_v3_large",
        "params": "5.4M",
        "description": "MobileNetV3-Large: 轻量高精度，移动端首选",
    },
    # ShuffleNet 系列
    "shufflenet_v2_x0_5": {
        "builder": build_shufflenet,
        "variant": "shufflenet_v2_x0_5",
        "params": "1.4M",
        "description": "ShuffleNetV2-0.5x: 超轻量，极快推理速度",
    },
    "shufflenet_v2_x1_0": {
        "builder": build_shufflenet,
        "variant": "shufflenet_v2_x1_0",
        "params": "2.3M",
        "description": "ShuffleNetV2-1.0x: 轻量，速度与精度平衡",
    },
}


def get_available_models() -> List[str]:
    """
    返回所有可用模型名称列表。
    Return list of all available model names.

    Returns:
        List[str]: 可用模型名称列表。List of available model names.
    """
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    根据模型名称构建模型（统一入口函数）。
    Build model by name (unified entry function).

    Args:
        model_name (str): 模型名称，可选值见 get_available_models()。
                          Model name, see get_available_models() for options.
        num_classes (int): 分类类别数量。Number of classification classes.
        pretrained (bool): 是否加载 ImageNet 预训练权重。
                           Whether to load ImageNet pretrained weights.

    Returns:
        nn.Module: 构建好的模型。Built model.

    Raises:
        ValueError: 当模型名称不在注册表中时。When model name is not in registry.

    Example:
        >>> model = build_model("resnet18", num_classes=10, pretrained=True)
        >>> print(model)
    """
    model_name = model_name.lower().strip()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )

    registry_entry = MODEL_REGISTRY[model_name]
    builder_fn = registry_entry["builder"]
    variant = registry_entry["variant"]

    # 调用对应的构建函数
    # Call the corresponding build function
    model = builder_fn(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
    )

    return model


class ModelFactory:
    """
    模型工厂类，提供面向对象的模型管理接口。
    Model factory class providing object-oriented model management interface.

    Example:
        >>> factory = ModelFactory(num_classes=10, pretrained=True)
        >>> model = factory.create("resnet18")
        >>> factory.print_model_info("resnet18")
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
    ) -> None:
        """
        初始化模型工厂。
        Initialize model factory.

        Args:
            num_classes (int): 分类类别数量。Number of classification classes.
            pretrained (bool): 是否使用预训练权重。Whether to use pretrained weights.
        """
        self.num_classes = num_classes
        self.pretrained = pretrained

    def create(self, model_name: str) -> nn.Module:
        """
        创建指定名称的模型。
        Create model with specified name.

        Args:
            model_name (str): 模型名称。Model name.

        Returns:
            nn.Module: 构建好的模型。Built model.
        """
        return build_model(
            model_name=model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        )

    @staticmethod
    def list_models() -> None:
        """打印所有可用模型信息。Print all available model information."""
        print("\n" + "=" * 60)
        print("Available Models / 可用模型列表")
        print("=" * 60)
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:<30} {info['params']:<10} {info['description']}")
        print("=" * 60 + "\n")

    @staticmethod
    def print_model_info(model_name: str) -> None:
        """
        打印指定模型的详细信息。
        Print detailed information for specified model.

        Args:
            model_name (str): 模型名称。Model name.
        """
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}")
            return
        info = MODEL_REGISTRY[model_name]
        print(f"\nModel: {model_name}")
        print(f"  Parameters: {info['params']}")
        print(f"  Description: {info['description']}\n")

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        统计模型可训练参数数量。
        Count number of trainable parameters in model.

        Args:
            model (nn.Module): PyTorch 模型。PyTorch model.

        Returns:
            int: 可训练参数总数。Total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)