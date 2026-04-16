"""
模型单元测试 / Model Unit Tests

测试所有支持的模型是否能正确构建和前向传播。
Test that all supported models can be correctly built and forward-passed.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model, get_available_models
from models.model_factory import ModelFactory


class TestModelFactory:
    """测试模型工厂。Test model factory."""

    def test_get_available_models(self):
        """测试获取可用模型列表。Test getting available models list."""
        models = get_available_models()
        assert len(models) > 0
        assert "resnet18" in models
        assert "mobilenet_v2" in models
        assert "shufflenet_v2_x1_0" in models

    def test_build_invalid_model(self):
        """测试构建无效模型时抛出异常。Test that building invalid model raises exception."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("invalid_model_name", num_classes=10)

    def test_model_factory_list(self, capsys):
        """测试模型工厂打印列表。Test model factory list printing."""
        ModelFactory.list_models()
        captured = capsys.readouterr()
        assert "resnet18" in captured.out

    def test_count_parameters(self):
        """测试参数计数。Test parameter counting."""
        model = build_model("resnet18", num_classes=10, pretrained=False)
        count = ModelFactory.count_parameters(model)
        assert count > 0


class TestResNet:
    """测试 ResNet 系列模型。Test ResNet series models."""

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_build(self, variant):
        """测试 ResNet 模型构建（不加载预训练权重）。Test ResNet model building (no pretrained)."""
        model = build_model(variant, num_classes=10, pretrained=False)
        assert model is not None

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_forward(self, variant):
        """测试 ResNet 前向传播输出形状。Test ResNet forward pass output shape."""
        model = build_model(variant, num_classes=10, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    @pytest.mark.parametrize("num_classes", [2, 10, 100, 1000])
    def test_resnet18_num_classes(self, num_classes):
        """测试 ResNet18 不同类别数输出。Test ResNet18 with different num_classes."""
        model = build_model("resnet18", num_classes=num_classes, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, num_classes)


class TestMobileNet:
    """测试 MobileNet 系列模型。Test MobileNet series models."""

    @pytest.mark.parametrize(
        "variant",
        ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"],
    )
    def test_mobilenet_build(self, variant):
        """测试 MobileNet 模型构建。Test MobileNet model building."""
        model = build_model(variant, num_classes=10, pretrained=False)
        assert model is not None

    @pytest.mark.parametrize(
        "variant",
        ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"],
    )
    def test_mobilenet_forward(self, variant):
        """测试 MobileNet 前向传播输出形状。Test MobileNet forward pass output shape."""
        model = build_model(variant, num_classes=10, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 10)


class TestShuffleNet:
    """测试 ShuffleNet 系列模型。Test ShuffleNet series models."""

    @pytest.mark.parametrize(
        "variant",
        ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0"],
    )
    def test_shufflenet_build(self, variant):
        """测试 ShuffleNet 模型构建。Test ShuffleNet model building."""
        model = build_model(variant, num_classes=10, pretrained=False)
        assert model is not None

    @pytest.mark.parametrize(
        "variant",
        ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0"],
    )
    def test_shufflenet_forward(self, variant):
        """测试 ShuffleNet 前向传播输出形状。Test ShuffleNet forward pass output shape."""
        model = build_model(variant, num_classes=10, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 10)