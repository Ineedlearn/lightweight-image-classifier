# 🔥 PyTorch Lightweight Image Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![CI](https://github.com/yourusername/lightweight-image-classifier/actions/workflows/ci.yml/badge.svg)

**一个开箱即用的 PyTorch 轻量化图像分类框架**

[English](#english) | [中文](#chinese)

</div>

---

<a name="chinese"></a>
## 📖 中文文档

### ✨ 特性

- 🚀 **多模型支持**：一键切换 ResNet(18/34/50)、MobileNetV2/V3、ShuffleNetV2
- 📦 **自定义数据集**：支持 ImageFolder 格式，内置丰富数据增强策略
- 📊 **训练可视化**：集成 TensorBoard，自动绘制准确率与损失曲线
- 🔄 **断点续训**：支持从 checkpoint 恢复训练
- ⚡ **混合精度**：支持 AMP 加速训练
- 📤 **模型导出**：支持导出为 ONNX 格式
- 🐳 **Docker 部署**：一键 Docker 部署，含 GPU 支持
- 🌐 **中英双语**：完整的中英文文档

### 📁 目录结构

```
lightweight-image-classifier/
├── configs/                    # 配置文件
│   └── default.yaml           # 默认配置
├── datasets/                   # 数据集模块
│   ├── __init__.py
│   ├── custom_dataset.py      # 自定义数据集类
│   └── transforms.py          # 数据增强策略
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── model_factory.py       # 模型工厂（统一接口）
│   ├── resnet.py              # ResNet 系列
│   ├── mobilenet.py           # MobileNet 系列
│   └── shufflenet.py          # ShuffleNet 系列
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── logger.py              # 日志工具
│   ├── metrics.py             # 评估指标
│   ├── checkpoint.py          # 检查点管理
│   └── visualizer.py          # 可视化工具
├── scripts/                    # 脚本
│   ├── train.py               # 训练脚本
│   ├── validate.py            # 验证脚本
│   ├── inference.py           # 推理脚本
│   └── export_onnx.py         # ONNX 导出脚本
├── tests/                      # 单元测试
│   ├── test_models.py
│   └── test_datasets.py
├── docker/                     # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/                    # GitHub Actions
│   └── workflows/
│       └── ci.yml
├── requirements.txt            # 依赖列表
├── setup.py                    # 安装配置
├── .gitignore
└── LICENSE
```

### 🚀 快速开始

#### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/yourusername/lightweight-image-classifier.git
cd lightweight-image-classifier

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 或者通过 setup.py 安装
pip install -e .
```

#### 2. 准备数据集

数据集需按照 ImageFolder 格式组织：

```
data/
├── train/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class_1/
│       ├── img003.jpg
│       └── img004.jpg
└── val/
    ├── class_0/
    └── class_1/
```

#### 3. 开始训练

```bash
# 使用 ResNet18 训练
python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10

# 使用 MobileNetV2 训练，开启混合精度
python scripts/train.py --model mobilenet_v2 --data_dir ./data --num_classes 10 --amp

# 使用 ShuffleNetV2 训练，指定批大小和学习率
python scripts/train.py --model shufflenet_v2 --data_dir ./data \
    --num_classes 10 --batch_size 64 --lr 0.01

# 从断点续训
python scripts/train.py --model resnet18 --data_dir ./data \
    --num_classes 10 --resume ./checkpoints/checkpoint_epoch_10.pth
```

#### 4. 验证模型

```bash
python scripts/validate.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data --num_classes 10
```

#### 5. 推理

```bash
# 单张图片推理
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image ./test.jpg --num_classes 10 --topk 5

# 批量推理
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image_dir ./test_images --num_classes 10
```

#### 6. 导出 ONNX

```bash
python scripts/export_onnx.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 10 --output model.onnx
```

#### 7. 查看训练曲线

```bash
tensorboard --logdir ./runs
```

### 🐳 Docker 部署

```bash
# CPU 版本
docker-compose up

# GPU 版本
docker-compose -f docker/docker-compose.yml up gpu-trainer
```

### 📊 支持的模型

| 模型 | 参数量 | Top-1 精度 (ImageNet) | 推理速度 |
|------|--------|----------------------|----------|
| ResNet18 | 11.7M | 69.8% | 快 |
| ResNet34 | 21.8M | 73.3% | 中 |
| ResNet50 | 25.6M | 76.1% | 中 |
| MobileNetV2 | 3.4M | 71.8% | 很快 |
| MobileNetV3-Small | 2.5M | 67.7% | 极快 |
| MobileNetV3-Large | 5.4M | 74.0% | 很快 |
| ShuffleNetV2-0.5x | 1.4M | 60.6% | 极快 |
| ShuffleNetV2-1.0x | 2.3M | 69.4% | 极快 |

### ⚙️ 配置说明

编辑 `configs/default.yaml` 可调整所有训练参数：

```yaml
model:
  name: resnet18          # 模型名称
  pretrained: true        # 是否使用预训练权重

training:
  epochs: 100             # 训练轮数
  batch_size: 32          # 批大小
  lr: 0.01               # 初始学习率
  amp: false             # 混合精度

data:
  num_classes: 10         # 分类数量
  input_size: 224         # 输入图像尺寸
```

---

<a name="english"></a>
## 📖 English Documentation

### ✨ Features

- 🚀 **Multi-Model Support**: One-click switch between ResNet(18/34/50), MobileNetV2/V3, ShuffleNetV2
- 📦 **Custom Dataset**: ImageFolder format support with rich data augmentation
- 📊 **Training Visualization**: TensorBoard integration, auto-plot accuracy & loss curves
- 🔄 **Resume Training**: Resume from checkpoint
- ⚡ **Mixed Precision**: AMP support for faster training
- 📤 **Model Export**: Export to ONNX format
- 🐳 **Docker Deploy**: One-click Docker deployment with GPU support
- 🌐 **Bilingual Docs**: Full Chinese & English documentation

### 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/lightweight-image-classifier.git
cd lightweight-image-classifier

# Install dependencies
pip install -r requirements.txt

# Train with ResNet18
python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10

# Validate
python scripts/validate.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data --num_classes 10

# Inference
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image ./test.jpg --num_classes 10
```

### 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ for the open-source community
</div>