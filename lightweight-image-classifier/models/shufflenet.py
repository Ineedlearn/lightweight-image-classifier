"""
ShuffleNet 系列模型模块 / ShuffleNet Series Model Module

支持 ShuffleNetV2-0.5x 和 ShuffleNetV2-1.0x，可选 ImageNet 预训练权重。
Supports ShuffleNetV2-0.5x and ShuffleNetV2-1.0x with optional pretrained weights.

参考论文 / Reference Paper:
    "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    Ma et al., ECCV 2018
    https://arxiv.org/abs/1807.11164
"""

import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
)


# ShuffleNet 变体配置表
# ShuffleNet variant configuration table
SHUFFLENET_CONFIGS = {
    "shufflenet_v2_x0_5": {
        "model_fn": tv_models.shufflenet_v2_x0_5,
        "weights": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
        "out_features": 1024,   # 0.5x 版本最后一层输出通道数
    },
    "shufflenet_v2_x1_0": {
        "model_fn": tv_models.shufflenet_v2_x1_0,
        "weights": ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        "out_features": 1024,   # 1.0x 版本最后一层输出通道数
    },
}


def build_shufflenet(
    variant: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    构建 ShuffleNetV2 模型并替换分类头。
    Build ShuffleNetV2 model and replace classification head.

    ShuffleNetV2 分类头结构：
        model.fc = Linear(in_features, 1000)
    直接替换 fc 层即可。
    Directly replace the fc layer.

    Args:
        variant (str): ShuffleNet 变体名称。ShuffleNet variant name.
        num_classes (int): 目标分类类别数。Target number of classes.
        pretrained (bool): 是否加载 ImageNet 预训练权重。Load pretrained weights.

    Returns:
        nn.Module: 修改后的 ShuffleNetV2 模型。Modified ShuffleNetV2 model.

    Raises:
        ValueError: 当 variant 不在支持列表中时。When variant is not supported.
    """
    if variant not in SHUFFLENET_CONFIGS:
        raise ValueError(
            f"Unsupported ShuffleNet variant: '{variant}'. "
            f"Choose from: {list(SHUFFLENET_CONFIGS.keys())}"
        )

    config = SHUFFLENET_CONFIGS[variant]

    # 加载预训练模型或随机初始化
    # Load pretrained model or random initialization
    if pretrained:
        model = config["model_fn"](weights=config["weights"])
    else:
        model = config["model_fn"](weights=None)

    # ShuffleNetV2 使用 model.fc 作为分类层，直接替换
    # ShuffleNetV2 uses model.fc as classifier, replace directly
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model