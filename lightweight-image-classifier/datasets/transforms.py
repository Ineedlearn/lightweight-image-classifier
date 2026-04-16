"""
数据增强策略模块 / Data Augmentation Transforms Module

提供训练集和验证集的数据预处理与增强策略。
Provides data preprocessing and augmentation strategies for train/val sets.
"""

from typing import Optional

import torchvision.transforms as T


# ImageNet 标准化参数（均值和标准差）
# ImageNet normalization parameters (mean and std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    input_size: int = 224,
    use_color_jitter: bool = True,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    flip_prob: float = 0.5,
    use_random_erasing: bool = False,
    erasing_prob: float = 0.2,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> T.Compose:
    """
    构建训练集数据增强流水线。
    Build training data augmentation pipeline.

    Args:
        input_size (int): 输入图像尺寸（正方形）。Input image size (square).
        use_color_jitter (bool): 是否使用颜色抖动。Whether to use color jitter.
        brightness (float): 亮度变化范围 [max(0, 1-b), 1+b]。Brightness jitter range.
        contrast (float): 对比度变化范围。Contrast jitter range.
        saturation (float): 饱和度变化范围。Saturation jitter range.
        hue (float): 色调变化范围 [-h, h]，h ∈ [0, 0.5]。Hue jitter range.
        flip_prob (float): 随机水平翻转概率。Random horizontal flip probability.
        use_random_erasing (bool): 是否使用随机擦除。Whether to use random erasing.
        erasing_prob (float): 随机擦除概率。Random erasing probability.
        mean (list, optional): 归一化均值，默认使用 ImageNet 均值。Normalization mean.
        std (list, optional): 归一化标准差，默认使用 ImageNet 标准差。Normalization std.

    Returns:
        T.Compose: 组合后的数据增强变换。Composed data augmentation transforms.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    # 构建增强操作列表
    # Build augmentation operations list
    transform_list = [
        # 1. 随机裁剪缩放：从随机位置和尺度裁剪后缩放到目标尺寸
        #    Random resized crop: crop from random position/scale then resize
        T.RandomResizedCrop(
            size=input_size,
            scale=(0.08, 1.0),    # 裁剪面积比例范围
            ratio=(3.0 / 4, 4.0 / 3),  # 裁剪宽高比范围
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        # 2. 随机水平翻转
        #    Random horizontal flip
        T.RandomHorizontalFlip(p=flip_prob),
    ]

    # 3. 可选：颜色抖动（亮度、对比度、饱和度、色调）
    #    Optional: color jitter (brightness, contrast, saturation, hue)
    if use_color_jitter:
        transform_list.append(
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    # 4. 转换为 Tensor 并归一化
    #    Convert to tensor and normalize
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # 5. 可选：随机擦除（在归一化后进行）
    #    Optional: random erasing (applied after normalization)
    if use_random_erasing:
        transform_list.append(
            T.RandomErasing(
                p=erasing_prob,
                scale=(0.02, 0.33),  # 擦除区域面积比例
                ratio=(0.3, 3.3),    # 擦除区域宽高比
                value=0,             # 填充值（0 表示黑色）
            )
        )

    return T.Compose(transform_list)


def get_val_transforms(
    input_size: int = 224,
    resize_size: Optional[int] = None,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> T.Compose:
    """
    构建验证/测试集数据预处理流水线（无随机增强）。
    Build validation/test data preprocessing pipeline (no random augmentation).

    Args:
        input_size (int): 最终裁剪尺寸。Final crop size.
        resize_size (int, optional): 先缩放到此尺寸，默认为 input_size * 256 // 224。
                                     Resize to this size first.
        mean (list, optional): 归一化均值。Normalization mean.
        std (list, optional): 归一化标准差。Normalization std.

    Returns:
        T.Compose: 组合后的预处理变换。Composed preprocessing transforms.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    # 验证集缩放尺寸：通常比裁剪尺寸稍大（如 256 -> 224）
    # Val resize size: usually slightly larger than crop size (e.g., 256 -> 224)
    if resize_size is None:
        resize_size = int(input_size * 256 / 224)

    return T.Compose([
        # 1. 先缩放短边到 resize_size
        #    Resize shorter edge to resize_size
        T.Resize(
            size=resize_size,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        # 2. 中心裁剪到目标尺寸
        #    Center crop to target size
        T.CenterCrop(size=input_size),
        # 3. 转换为 Tensor 并归一化
        #    Convert to tensor and normalize
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    input_size: int = 224,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> T.Compose:
    """
    构建推理用的数据预处理流水线（与验证集相同）。
    Build inference data preprocessing pipeline (same as validation).

    Args:
        input_size (int): 输入图像尺寸。Input image size.
        mean (list, optional): 归一化均值。Normalization mean.
        std (list, optional): 归一化标准差。Normalization std.

    Returns:
        T.Compose: 推理预处理变换。Inference preprocessing transforms.
    """
    return get_val_transforms(input_size=input_size, mean=mean, std=std)