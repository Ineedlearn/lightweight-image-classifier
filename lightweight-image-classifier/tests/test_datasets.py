"""
数据集单元测试 / Dataset Unit Tests

测试数据集类和数据增强变换的正确性。
Test correctness of dataset classes and data augmentation transforms.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_inference_transforms,
    get_train_transforms,
    get_val_transforms,
)


def create_dummy_dataset(root: Path, num_classes: int = 3, images_per_class: int = 5):
    """
    创建临时测试数据集（随机图像）。
    Create temporary test dataset (random images).

    Args:
        root (Path): 数据集根目录。Dataset root directory.
        num_classes (int): 类别数量。Number of classes.
        images_per_class (int): 每类图像数量。Images per class.
    """
    for cls_idx in range(num_classes):
        cls_dir = root / f"class_{cls_idx}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(images_per_class):
            # 创建随机 RGB 图像
            # Create random RGB image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(cls_dir / f"img_{img_idx:04d}.jpg")


class TestTransforms:
    """测试数据增强变换。Test data augmentation transforms."""

    def test_train_transforms_output_shape(self):
        """测试训练变换输出形状。Test training transform output shape."""
        transform = get_train_transforms(input_size=224)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        output = transform(dummy_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_val_transforms_output_shape(self):
        """测试验证变换输出形状。Test validation transform output shape."""
        transform = get_val_transforms(input_size=224)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        output = transform(dummy_image)
        assert output.shape == (3, 224, 224)

    def test_inference_transforms_output_shape(self):
        """测试推理变换输出形状。Test inference transform output shape."""
        transform = get_inference_transforms(input_size=224)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        output = transform(dummy_image)
        assert output.shape == (3, 224, 224)

    def test_train_transforms_with_color_jitter(self):
        """测试带颜色抖动的训练变换。Test training transforms with color jitter."""
        transform = get_train_transforms(input_size=224, use_color_jitter=True)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        output = transform(dummy_image)
        assert output.shape == (3, 224, 224)

    def test_train_transforms_with_random_erasing(self):
        """测试带随机擦除的训练变换。Test training transforms with random erasing."""
        transform = get_train_transforms(
            input_size=224,
            use_random_erasing=True,
            erasing_prob=1.0,  # 确保擦除一定发生
        )
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        output = transform(dummy_image)
        assert output.shape == (3, 224, 224)

    def test_custom_input_size(self):
        """测试自定义输入尺寸。Test custom input size."""
        for size in [112, 160, 224, 299]:
            transform = get_val_transforms(input_size=size)
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            )
            output = transform(dummy_image)
            assert output.shape == (3, size, size)


class TestClassificationDataset:
    """测试自定义分类数据集。Test custom classification dataset."""

    def test_dataset_creation(self):
        """测试数据集创建。Test dataset creation."""
        from datasets.custom_dataset import ClassificationDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_dummy_dataset(root, num_classes=3, images_per_class=5)

            dataset = ClassificationDataset(root=str(root))
            assert len(dataset) == 15  # 3 classes * 5 images
            assert len(dataset.classes) == 3

    def test_dataset_getitem(self):
        """测试数据集 __getitem__。Test dataset __getitem__."""
        from datasets.custom_dataset import ClassificationDataset
        from datasets.transforms import get_val_transforms

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_dummy_dataset(root, num_classes=3, images_per_class=5)

            transform = get_val_transforms(input_size=224)
            dataset = ClassificationDataset(root=str(root), transform=transform)

            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 224, 224)
            assert isinstance(label, int)
            assert 0 <= label < 3

    def test_dataset_class_distribution(self):
        """测试类别分布统计。Test class distribution statistics."""
        from datasets.custom_dataset import ClassificationDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_dummy_dataset(root, num_classes=3, images_per_class=5)

            dataset = ClassificationDataset(root=str(root))
            distribution = dataset.get_class_distribution()

            assert len(distribution) == 3
            for count in distribution.values():
                assert count == 5

    def test_dataset_invalid_dir(self):
        """测试无效目录时抛出异常。Test exception on invalid directory."""
        from datasets.custom_dataset import ClassificationDataset

        with pytest.raises(FileNotFoundError):
            ClassificationDataset(root="/nonexistent/path")