"""
日志管理模块
使用 loguru 提供更好的日志管理功能
"""

import os
import sys
import logging
from pathlib import Path
from loguru import logger
from datetime import datetime
import gzip
import shutil


class LogManager:
    """日志管理器 - 基于 loguru 的高性能日志管理"""

    def __init__(self, config):
        """
        初始化日志管理器

        Args:
            config: 配置对象
        """
        self.config = config
        # 处理日志目录路径
        self.log_dir = config.LOG_PATH

        # 如果路径是空的，使用当前目录
        if not self.log_dir:
            self.log_dir = "."

        # 设置主要日志文件路径
        self.main_log_file = os.path.join(self.log_dir, "app.log")

        # 确保日志目录存在
        self._ensure_log_directory()

        # 配置 loguru
        self._setup_loguru()

        # 清理旧日志
        self._cleanup_old_logs()

    def _ensure_log_directory(self):
        """确保日志目录存在"""
        # 只有在需要创建非当前目录时才执行
        if self.log_dir and self.log_dir != '.':
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except PermissionError as e:
                # 如果权限不足，回退到当前目录
                print(f"Warning: Cannot create log directory {self.log_dir}: {e}")
                print(f"Falling back to current directory")
                self.log_dir = "."
                self.main_log_file = os.path.join(self.log_dir, "app.log")

    def _setup_loguru(self):
        """配置 loguru 日志系统"""

        # 移除默认处理器
        logger.remove()

        # 1. 控制台输出（从配置文件获取日志级别）
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>Thread-{thread.id}</yellow> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.config.LOG_LEVEL,
            colorize=True
        )

        # 2. 主日志文件 - 按大小轮转
        logger.add(
            self.main_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | Thread-{thread.id} | {name}:{function}:{line} - {message}",
            level=self.config.LOG_LEVEL,
            rotation=getattr(self.config, 'LOG_MAX_SIZE_MB', 10) * 1024 * 1024,  # 按大小轮转
            retention=getattr(self.config, 'LOG_BACKUP_COUNT', 5),               # 保留文件数
            compression="gz",  # 自动压缩旧文件
            encoding="utf-8"
        )

        # 3. 错误日志文件 - 单独记录错误
        error_log_file = os.path.join(self.log_dir, "app_error.log")
        logger.add(
            error_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | Thread-{thread.id} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="10 MB",
            retention=7,
            compression="gz",
            encoding="utf-8"
        )

        # 4. 访问日志文件 - 记录 HTTP 请求
        access_log_file = os.path.join(self.log_dir, "app_access.log")
        logger.add(
            access_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | Thread-{thread.id} | {message}",
            level="DEBUG",
            rotation="daily",
            retention=getattr(self.config, 'LOG_DAILY_BACKUP_COUNT', 7),
            compression="gz",
            encoding="utf-8",
            filter=lambda record: "access" in record["extra"]
        )

        # 5. 调试日志文件 - 仅在 DEBUG 模式下启用
        if self.config.LOG_LEVEL == "DEBUG":
            debug_log_file = os.path.join(self.log_dir, "app_debug.log")
            logger.add(
                debug_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | Thread-{thread.id} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="50 MB",
                retention=3,  # 调试日志保留较少
                compression="gz",
                encoding="utf-8"
            )

    def _cleanup_old_logs(self):
        """清理超过指定天数的日志文件"""
        if not os.path.exists(self.log_dir):
            return

        max_age_days = getattr(self.config, 'LOG_MAX_AGE_DAYS', 30)
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 60 * 60

        cleaned_count = 0
        for filename in os.listdir(self.log_dir):
            if filename.endswith(('.log', '.gz')):
                file_path = os.path.join(self.log_dir, filename)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
                except (OSError, IOError) as e:
                    logger.warning(f"Error cleaning up log file {filename}: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old log files")

    def get_logger(self, name=None):
        """
        获取日志记录器

        Args:
            name: 日志记录器名称

        Returns:
            logger: loguru 日志记录器
        """
        if name:
            return logger.bind(name=name)
        return logger

    def log_access(self, message):
        """记录访问日志"""
        logger.bind(access=True).info(message)

    def log_performance(self, operation, duration, **kwargs):
        """记录性能日志"""
        perf_msg = f"Performance: {operation} took {duration:.3f}s"
        if kwargs:
            perf_msg += f" | {', '.join([f'{k}={v}' for k, v in kwargs.items()])}"
        logger.info(perf_msg)

    def log_error_with_context(self, error, context=None):
        """记录带上下文的错误日志"""
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        logger.error(error_msg)

    def get_log_stats(self):
        """获取日志统计信息"""
        if not os.path.exists(self.log_dir):
            return {"error": "Log directory not found"}

        log_files = []
        total_size = 0

        for filename in os.listdir(self.log_dir):
            if filename.endswith(('.log', '.gz')):
                file_path = os.path.join(self.log_dir, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    file_mtime = os.path.getmtime(file_path)
                    file_age_days = (datetime.now().timestamp() - file_mtime) / (24 * 60 * 60)

                    log_files.append({
                        "filename": filename,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "age_days": round(file_age_days, 1),
                        "modified": datetime.fromtimestamp(file_mtime).isoformat(),
                        "compressed": filename.endswith('.gz')
                    })
                    total_size += file_size
                except (OSError, IOError) as e:
                    logger.warning(f"Error reading log file {filename}: {e}")

        # 按修改时间排序
        log_files.sort(key=lambda x: x['modified'], reverse=True)

        return {
            "log_directory": self.log_dir,
            "total_files": len(log_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": log_files
        }


def setup_logging(config):
    """
    设置日志系统

    Args:
        config: 配置对象

    Returns:
        LogManager: 日志管理器实例
    """
    # 创建日志管理器
    log_manager = LogManager(config)

    # 获取根日志记录器
    root_logger = log_manager.get_logger()

    # 禁用标准库的 logging，避免重复输出
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())

    return log_manager, root_logger


# 兼容性函数
def get_logger(name=None):
    """获取日志记录器（兼容性函数）"""
    return logger.bind(name=name) if name else logger
