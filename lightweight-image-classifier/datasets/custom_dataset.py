"""
自定义数据集模块 / Custom Dataset Module

提供基于 ImageFolder 格式的自定义数据集类和 DataLoader 创建函数。
Provides custom dataset class based on ImageFolder format and DataLoader creation utilities.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from datasets.transforms import get_train_transforms, get_val_transforms


class ClassificationDataset(Dataset):
    """
    自定义图像分类数据集类，支持 ImageFolder 格式。
    Custom image classification dataset supporting ImageFolder format.

    目录结构 / Directory structure:
        root/
        ├── class_0/
        │   ├── img001.jpg
        │   └── img002.jpg
        └── class_1/
            ├── img003.jpg
            └── img004.jpg

    Args:
        root (str): 数据集根目录路径。Root directory path of the dataset.
        transform (callable, optional): 数据增强/预处理变换。Data augmentation/preprocessing transforms.
        extensions (tuple): 支持的图像文件扩展名。Supported image file extensions.
    """

    # 支持的图像格式
    # Supported image formats
    SUPPORTED_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".bmp",
        ".gif", ".tiff", ".tif", ".webp",
    )

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS

        # 验证目录存在
        # Validate directory exists
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        # 扫描类别目录，构建类别到索引的映射
        # Scan class directories, build class-to-index mapping
        self.classes, self.class_to_idx = self._find_classes()

        # 扫描所有图像文件，构建 (image_path, label) 列表
        # Scan all image files, build (image_path, label) list
        self.samples = self._make_dataset()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid images found in {self.root}. "
                f"Supported extensions: {self.extensions}"
            )

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        扫描根目录下的子目录作为类别。
        Scan subdirectories under root as classes.

        Returns:
            Tuple[List[str], Dict[str, int]]: 类别列表和类别到索引的映射。
                                              Class list and class-to-index mapping.
        """
        # 获取所有子目录（排除隐藏目录）
        # Get all subdirectories (excluding hidden ones)
        classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not classes:
            raise RuntimeError(
                f"No class directories found in {self.root}. "
                "Please organize data in ImageFolder format."
            )

        # 构建类别名到整数索引的映射
        # Build class name to integer index mapping
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[Path, int]]:
        """
        扫描所有类别目录，收集图像路径和对应标签。
        Scan all class directories, collect image paths and corresponding labels.

        Returns:
            List[Tuple[Path, int]]: (图像路径, 类别索引) 列表。
                                    List of (image_path, class_index) tuples.
        """
        samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            for img_path in sorted(class_dir.iterdir()):
                # 只收集支持格式的图像文件
                # Only collect image files with supported extensions
                if img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, class_idx))
        return samples

    def __len__(self) -> int:
        """返回数据集样本总数。Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        获取指定索引的样本。
        Get sample at specified index.

        Args:
            index (int): 样本索引。Sample index.

        Returns:
            Tuple[torch.Tensor, int]: (图像张量, 类别索引)。(image tensor, class index).
        """
        img_path, label = self.samples[index]

        # 加载图像并转换为 RGB（确保三通道）
        # Load image and convert to RGB (ensure 3 channels)
        image = Image.open(img_path).convert("RGB")

        # 应用数据增强/预处理变换
        # Apply data augmentation/preprocessing transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_names(self) -> List[str]:
        """返回类别名称列表。Return list of class names."""
        return self.classes

    def get_class_distribution(self) -> Dict[str, int]:
        """
        统计各类别的样本数量。
        Count number of samples per class.

        Returns:
            Dict[str, int]: 类别名称到样本数量的映射。Class name to sample count mapping.
        """
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution


def create_dataloaders(
    data_dir: str,
    num_classes: int,
    input_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_color_jitter: bool = True,
    use_random_erasing: bool = False,
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    创建训练集和验证集的 DataLoader。
    Create DataLoaders for training and validation sets.

    Args:
        data_dir (str): 数据集根目录，需含 train/ 和 val/ 子目录。
                        Dataset root directory with train/ and val/ subdirectories.
        num_classes (int): 类别数量（用于验证）。Number of classes (for validation).
        input_size (int): 输入图像尺寸。Input image size.
        batch_size (int): 批大小。Batch size.
        num_workers (int): DataLoader 工作进程数。Number of DataLoader workers.
        pin_memory (bool): 是否固定内存（GPU 训练时建议开启）。Whether to pin memory.
        use_color_jitter (bool): 是否使用颜色抖动增强。Whether to use color jitter.
        use_random_erasing (bool): 是否使用随机擦除增强。Whether to use random erasing.
        train_subdir (str): 训练集子目录名。Training set subdirectory name.
        val_subdir (str): 验证集子目录名。Validation set subdirectory name.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]:
            (训练 DataLoader, 验证 DataLoader, 类别名称列表)。
            (train DataLoader, val DataLoader, class names list).
    """
    train_dir = os.path.join(data_dir, train_subdir)
    val_dir = os.path.join(data_dir, val_subdir)

    # 构建训练集（含数据增强）
    # Build training set (with data augmentation)
    train_transform = get_train_transforms(
        input_size=input_size,
        use_color_jitter=use_color_jitter,
        use_random_erasing=use_random_erasing,
    )
    train_dataset = ClassificationDataset(
        root=train_dir,
        transform=train_transform,
    )

    # 构建验证集（仅做中心裁剪和归一化）
    # Build validation set (only center crop and normalization)
    val_transform = get_val_transforms(input_size=input_size)
    val_dataset = ClassificationDataset(
        root=val_dir,
        transform=val_transform,
    )

    # 验证类别数量是否匹配
    # Validate class count matches
    assert len(train_dataset.classes) == num_classes, (
        f"Expected {num_classes} classes, "
        f"but found {len(train_dataset.classes)} in {train_dir}"
    )

    # 创建 DataLoader
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练集打乱顺序
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # 丢弃最后不完整的批次（保持批大小一致）
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,          # 验证集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset.classes