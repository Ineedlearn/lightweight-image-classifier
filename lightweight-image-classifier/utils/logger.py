"""
日志工具模块 / Logger Utility Module

提供统一的日志配置，同时输出到控制台和文件。
Provides unified logging configuration with output to both console and file.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "image_classifier",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    配置并返回一个日志记录器。
    Configure and return a logger.

    日志格式示例 / Log format example:
        2024-01-01 12:00:00 | INFO | train.py:100 | Epoch [1/100] Loss: 0.5234

    Args:
        name (str): 日志记录器名称。Logger name.
        log_dir (str, optional): 日志文件保存目录，None 表示不保存文件。
                                 Log file directory, None means no file output.
        log_level (int): 日志级别，默认 INFO。Log level, default INFO.
        log_to_console (bool): 是否输出到控制台。Whether to output to console.
        log_to_file (bool): 是否输出到文件。Whether to output to file.

    Returns:
        logging.Logger: 配置好的日志记录器。Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加 handler（多次调用时）
    # Avoid adding duplicate handlers (when called multiple times)
    if logger.handlers:
        logger.handlers.clear()

    # 定义日志格式
    # Define log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 添加控制台 Handler
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件 Handler
    # Add file handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 以时间戳命名日志文件，避免覆盖
        # Name log file with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file: {log_file}")

    return logger


def get_logger(name: str = "image_classifier") -> logging.Logger:
    """
    获取已配置的日志记录器（若不存在则创建默认配置）。
    Get configured logger (create with default config if not exists).

    Args:
        name (str): 日志记录器名称。Logger name.

    Returns:
        logging.Logger: 日志记录器。Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 若未配置，创建一个仅输出到控制台的默认 logger
        # If not configured, create a default logger outputting to console only
        return setup_logger(name=name, log_to_file=False)
    return logger