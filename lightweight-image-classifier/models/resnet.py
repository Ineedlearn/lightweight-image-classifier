"""
ResNet 系列模型模块 / ResNet Series Model Module

支持 ResNet-18、ResNet-34、ResNet-50，可选 ImageNet 预训练权重。
Supports ResNet-18, ResNet-34, ResNet-50 with optional ImageNet pretrained weights.

参考论文 / Reference Paper:
    "Deep Residual Learning for Image Recognition"
    He et al., CVPR 2016
    https://arxiv.org/abs/1512.03385
"""

import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


# ResNet 变体配置表
# ResNet variant configuration table
RESNET_CONFIGS = {
    "resnet18": {
        "model_fn": tv_models.resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "out_features": 512,    # 最后一个卷积层输出通道数
    },
    "resnet34": {
        "model_fn": tv_models.resnet34,
        "weights": ResNet34_Weights.IMAGENET1K_V1,
        "out_features": 512,
    },
    "resnet50": {
        "model_fn": tv_models.resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V1,
        "out_features": 2048,   # ResNet-50 使用 Bottleneck，输出 2048
    },
}


def build_resnet(
    variant: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    构建 ResNet 模型并替换分类头。
    Build ResNet model and replace classification head.

    核心思路：加载预训练的 ResNet 骨干网络，将最后的全连接层（fc）
    替换为适配目标类别数的新线性层，实现迁移学习。

    Core idea: Load pretrained ResNet backbone, replace the final
    fully connected layer (fc) with a new linear layer adapted to
    the target number of classes, enabling transfer learning.

    Args:
        variant (str): ResNet 变体名称，如 'resnet18'。ResNet variant name.
        num_classes (int): 目标分类类别数。Target number of classes.
        pretrained (bool): 是否加载 ImageNet 预训练权重。Load pretrained weights.

    Returns:
        nn.Module: 修改后的 ResNet 模型。Modified ResNet model.

    Raises:
        ValueError: 当 variant 不在支持列表中时。When variant is not supported.
    """
    if variant not in RESNET_CONFIGS:
        raise ValueError(
            f"Unsupported ResNet variant: '{variant}'. "
            f"Choose from: {list(RESNET_CONFIGS.keys())}"
        )

    config = RESNET_CONFIGS[variant]

    # 加载预训练模型或随机初始化
    # Load pretrained model or random initialization
    if pretrained:
        model = config["model_fn"](weights=config["weights"])
    else:
        model = config["model_fn"](weights=None)

    # 替换最后的全连接层以适配目标类别数
    # Replace the final fully connected layer for target classes
    # 原始 fc 层：Linear(in_features, 1000)  ->  新 fc 层：Linear(in_features, num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model