"""
MobileNet 系列模型模块 / MobileNet Series Model Module

支持 MobileNetV2、MobileNetV3-Small、MobileNetV3-Large，可选 ImageNet 预训练权重。
Supports MobileNetV2, MobileNetV3-Small, MobileNetV3-Large with optional pretrained weights.

参考论文 / Reference Papers:
    MobileNetV2: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
                 Sandler et al., CVPR 2018
    MobileNetV3: "Searching for MobileNetV3"
                 Howard et al., ICCV 2019
"""

import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)


# MobileNet 变体配置表
# MobileNet variant configuration table
MOBILENET_CONFIGS = {
    "mobilenet_v2": {
        "model_fn": tv_models.mobilenet_v2,
        "weights": MobileNet_V2_Weights.IMAGENET1K_V1,
        "classifier_type": "v2",    # 分类头类型标识
    },
    "mobilenet_v3_small": {
        "model_fn": tv_models.mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "classifier_type": "v3",
    },
    "mobilenet_v3_large": {
        "model_fn": tv_models.mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "classifier_type": "v3",
    },
}


def build_mobilenet(
    variant: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    构建 MobileNet 模型并替换分类头。
    Build MobileNet model and replace classification head.

    MobileNetV2 分类头结构：
        Sequential(Dropout, Linear(1280, 1000))
    MobileNetV3 分类头结构：
        Sequential(Linear(960, 1280), Hardswish, Dropout, Linear(1280, 1000))

    我们只替换最后一个 Linear 层，保留 Dropout 等其他层。
    We only replace the last Linear layer, keeping Dropout and other layers.

    Args:
        variant (str): MobileNet 变体名称。MobileNet variant name.
        num_classes (int): 目标分类类别数。Target number of classes.
        pretrained (bool): 是否加载 ImageNet 预训练权重。Load pretrained weights.

    Returns:
        nn.Module: 修改后的 MobileNet 模型。Modified MobileNet model.

    Raises:
        ValueError: 当 variant 不在支持列表中时。When variant is not supported.
    """
    if variant not in MOBILENET_CONFIGS:
        raise ValueError(
            f"Unsupported MobileNet variant: '{variant}'. "
            f"Choose from: {list(MOBILENET_CONFIGS.keys())}"
        )

    config = MOBILENET_CONFIGS[variant]

    # 加载预训练模型或随机初始化
    # Load pretrained model or random initialization
    if pretrained:
        model = config["model_fn"](weights=config["weights"])
    else:
        model = config["model_fn"](weights=None)

    # 根据版本替换分类头中的最后一个线性层
    # Replace the last linear layer in classifier based on version
    if config["classifier_type"] == "v2":
        # MobileNetV2: model.classifier = [Dropout, Linear(1280, 1000)]
        # 获取原始 Linear 层的输入特征数
        # Get input features of original Linear layer
        in_features = model.classifier[-1].in_features
        # 替换最后一个 Linear 层
        # Replace the last Linear layer
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif config["classifier_type"] == "v3":
        # MobileNetV3: model.classifier = [Linear, Hardswish, Dropout, Linear(1280, 1000)]
        # 同样只替换最后一个 Linear 层
        # Also only replace the last Linear layer
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model