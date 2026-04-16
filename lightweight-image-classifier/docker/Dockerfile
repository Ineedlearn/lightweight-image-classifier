# ============================================================
# Dockerfile for PyTorch Lightweight Image Classifier
# ============================================================
# 多阶段构建：先安装依赖，再复制代码，减小镜像体积
# Multi-stage build: install deps first, then copy code, reduce image size

# 基础镜像：PyTorch 官方镜像（含 CUDA 支持）
# Base image: Official PyTorch image (with CUDA support)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
# Set working directory
WORKDIR /app

# 设置环境变量
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（OpenCV 需要）
# Install system dependencies (required by OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt（利用 Docker 层缓存）
# Copy requirements.txt first (leverage Docker layer caching)
COPY requirements.txt .

# 安装 Python 依赖
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
# Copy project code
COPY . .

# 安装项目本身（可编辑模式）
# Install the project itself (editable mode)
RUN pip install -e .

# 创建必要目录
# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/runs

# 暴露 TensorBoard 端口
# Expose TensorBoard port
EXPOSE 6006

# 默认命令：打印使用说明
# Default command: print usage instructions
CMD ["python", "-c", "\
import sys; \
sys.path.insert(0, '/app'); \
from models import get_available_models; \
print('PyTorch Lightweight Image Classifier'); \
print('Available models:', get_available_models()); \
print('Usage: python scripts/train.py --help'); \
"]