# lightweight-image-classifier
<div align="center">

# 🔥 PyTorch Lightweight Image Classifier

**一个开箱即用的 PyTorch 轻量化图像分类框架**
**A Ready-to-Use PyTorch Lightweight Image Classification Framework**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen?logo=github-actions)](https://github.com/Ineedlearn/lightweight-image-classifier/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker/)
[![ONNX](https://img.shields.io/badge/ONNX-Export-blueviolet?logo=onnx)](scripts/export_onnx.py)

[English](#-english) | [中文](#-中文)

</div>

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🚀 **多模型支持** | 一键切换 ResNet / MobileNet / SuffleNet，共 8 种架构 |
| 📦 **自定义数据集** | 支持 ImageFlder 格式，内置丰富数据增强策略 |
| 📊 **训练可视化** | 集成 TesorBoard，自动绘制损失/准确率曲线 |
| 🔄 **断点训** | 支持从任意 checkpoint 恢复训练 |
| ⚡ **混合精度** | AMP 自动混合精度，显著加速训练 |
| 📤 **模型导出** |一键导出 ONNX，跨平台部署 |
| 🐳 **Docker 部署** | 一键 Docker 部署，含 GPU 支持 |
| ✅ **单元测试**| 34 项测试全部通过，代码质量有保障 |

---

## 📁 项目结构

```
lightweight-image-classiier/
├── 📂 configs/
│   └── default.yaml           # 完整训练配置（可 YAML 一键调参）
├── 📂 datasets/
│   ├── custom_dataset.py       # 自定义 Dataset 类（ImageFolder 格式）
│   └── transforms.py          # 训练 / 验证 / 推理数据增强流水线
├── 📂 models/
│  ├── model_factory.py        # 统一模型工厂（一键切换任意模型）
│   ├── resnet.py               # ResNet-18 / 34 / 50
│   ├── mobilenet.py            # MobileNetV2 / V3-Small / V3-Large
│   └── shufflenet.py           # ShuffleNetV2-0.5x / 1.0x
├── 📂 utils/
│   ├── logger.py              # 日志工具（控制台 + 文件双输出）
│   ├── metrics.py              # Top-K 准确率、混淆矩阵
│   ├── checkpoint.py           # 断点续训、自动清理旧检查点
│   └── visualizer.py           # 训练曲线、混淆矩阵热力图
├── 📂 scripts/
│   ├── train.py               # 完整训练脚本（AMP / 断点续训 / TensorBoard）
│   ├── validate.py             # 验证脚本（含混淆矩阵输出）
│   ├── inference.py            # 推理脚本（单张 / 批量 / Top-K）
│   └── export_onnx.py          # ONNX 导出 + 自动验证
├── 📂 tests/
│   ├── test_models.py          # 模型元测试（34 项全通过 ✅）
│   └── test_datasets.py        # 数据集单元测试
├── 📂 docker/
│   ├── Dockerfile             # 多阶段构建，含 CUDA 支持
│   └── docker-compose.yml      # CPU / GPU / TensorBoard 一键启动
├── 📂 .github/workflows/
│   └── ci.yml                  # GitHub Actions 自动化 CI
├── requirements.txt
├── setup.py
└── LICENSE                     # MIT 开源协议
```

---

## 🚀 快速

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/Ineedlearn/lightweight-image-classifier.git
cd lightweight-image-classifier

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

按照 **ImageFolder** 格式组织数据：

```
data/
├── train/
│   ├── cat/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── dog/
│       ├── 003.jpg
│       └── 004.jpg
└── val/
    ├── cat/
    └── dog/
```

### 3. 开始训练

```bash
# 使用 ResNet18 训练（最快上手）
python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10

# 使用 MobileNetV2 + 混合精度（推荐 GPU 用户）
python scripts/train.py --model mobilenet_v2 --data_dir ./data \
    --num_classes 10 --amp --batch_size 64

# 从断点续训
python scripts/train.py --model resnet18 --data_dir ./data \
    --num_classes 10 --resume ./checkpoints/checkpoint_epoch_0010.pth
```

### 4. 验证 & 推理

```bash
# 验证模型（输出准确率 + 混淆矩阵）
python scripts/validate.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data --num_classes 10

# 单张图片推理（输出 Top-5）
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image ./test.jpg --num_classes 10 --topk 5

# 批量推理（整个文件夹）
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image_dir ./test_images --num_classes 10 \
    --output ./results/predictions.json
```

### 5. 导出 ONNX

```bash
python scripts/export_onnx.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 10 --output model.onnx
```

### 6. 查看训练曲线

```bash
tensorboard --logdir ./runs
# 浏览器打开 http://localhost:6006
```

---

## 📊 支持的模型

| 模型 | 参数量 | ImageNet Top-1 | 适用场景 |
|------|--------|----------------|----------|
| `resne18` | 11.7M | 69.8% | 快速实验、基线对比 |
| `resnet34` | 21.8M | 73.3% | 精度与速度均衡 |
| `resnet50` | 25.6M | 76.1% | 高精度生产部署 |
| `mobilenet_v2` | 3.4M | 71.8% | 移动端 / 嵌入式 |
| `mobilenet_v3_small` | 2.5M | 67.7% | 极致轻量，边缘设备 |
| `mobilenet_v3_large` | 5.4M | 74.0% | 移动端首选 |
| `shufflenet_v2_x0_5` | 1.4M | 60.6% | 超轻量，极快推理 |
| `shufflenet_v2_x1_0` | 2.3M | 69.4% | 轻量速度精度平衡 |

> 一键切换模型，只需修改 `--model` 参数即可！

---

## ⚙️ 配置文件

编辑 `configs/default.yaml` 可调整所有训练参数，无需修改代码：

```yaml
model:
  name: resnet18          # 模型名称
  pretrained: true        # ImageNet 预训练权重

training:
  epochs: 100
  batch_size: 32
  lr: 0.01
  amp: false              # 混合精度

scheduler:
  name: cosine            # step / cosine / plateau

augmentation:
  train:
    random_horizontal_flip: true
    color_jitter: true
    random_erasing: false
```

---

## 🐳 Docker 一键部署

```bash
# CPU 训练
docker-compose -f docker/docker-compose.ml up cpu-trainer

# GPU 训练（需要 NVIDIA Docker）
docker-compose -f docker/docker-compose.yml up gpu-trainer

# 启动 TensorBoard
docker-compose -f docker/docker-compose.yml up tensorboard
```

---

## 🧪 运行测试

```bash
# 运行全部单元测试（34 项）
pytest tests/-v

# 查看测试覆盖率
pytest tests/ --cov=. --cov-report=html
```

---

## 📋 命令行参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `resnet18` | 模型名称 |
| `--data_dir`| — | 数据集根目录（必填）|
| `--num_classes` | — | 类别数量（必填）|
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `0.01` | 初始学习率 |
| `--amp` | `False` | 开启混合精度 |
| `--resume` | `None` | 断点续训路径 |
| `--scheduler` | `cosine` | 学习率调度策略 |
| `--gpu` | `0` | GPU 编号（-1=CPU）|

---

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 源，欢迎自由使用、修改和分发。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'feat: add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

---

<div align="center">

**如果这个项目对你有帮助，请点个 ⭐ Star 支持一下！**

Made with ❤️ for the open-source community

</div>

---

<a name="-english"></a>
## 🌐 English

### Quick Start

```bash
git clone https://github.com/Ineedlearn/lightweight-image-classifier.git
cd lightweight-image-classifier
pip install -r requirements.txt

# Train with ResNet18
python scripts/train.py --model resnet18 --data_dir ./data --num_classes 10

# Train with MobileNetV2 + AMP
python scripts/train.py --model mobilenet_v2 --data_dir ./data --num_classes 10 --amp

# Validate
python scripts/validate.py --modelresnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data --num_classes 10

# Inference (Top-5)
python scripts/inference.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --image ./test.jpg --num_classes 10 --topk 5

# Export to ONNX
python scripts/export_onnx.py --model resnet18 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 10 --output model.onnx
```

### Supported Models

8 models supported: ResNet-18/34/50, MobileNetV2, MobileNetV3-Small/Large, ShuffleNetV2-0.5x/1.0x.
Switch models with a single `--model` argument.

### License

[MIT License](LICENSE) — free to use, modify, and distribute.
