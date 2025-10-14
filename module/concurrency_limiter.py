"""
并发控制模块
限制同时处理的请求数量，保护低配置服务器资源
"""

import threading
import time
import logging
from queue import Queue, Empty, Full
from functools import wraps
from flask import jsonify

logger = logging.getLogger(__name__)


class ConcurrencyLimiter:
    """并发限制器"""

    def __init__(self, config):
        """
        初始化并发限制器

        Args:
            config: 配置对象
        """
        self.config = config
        self.max_concurrent = config.MAX_CONCURRENT_REQUESTS
        self.max_queue_size = config.MAX_QUEUE_SIZE
        self.request_timeout = config.REQUEST_TIMEOUT
        self.enabled = config.ENABLE_CONCURRENCY_LIMIT

        # 当前活跃的请求数
        self.active_requests = 0
        self.active_lock = threading.Lock()

        # 等待队列（使用信号量实现）
        self.semaphore = threading.Semaphore(self.max_concurrent)
        self.queue = Queue(maxsize=self.max_queue_size) if self.max_queue_size > 0 else None

        # 统计信息
        self.total_requests = 0
        self.rejected_requests = 0
        self.queued_requests = 0
        self.completed_requests = 0

        logger.info(f"并发限制器初始化: 最大并发={self.max_concurrent}, 队列大小={self.max_queue_size}")

    def acquire(self, timeout=None):
        """
        获取执行权限

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成功获取权限
        """
        if not self.enabled:
            return True

        timeout = timeout or self.request_timeout

        # 尝试获取信号量
        acquired = self.semaphore.acquire(blocking=True, timeout=timeout)

        if acquired:
            with self.active_lock:
                self.active_requests += 1
            logger.debug(f"请求获取执行权限 (活跃: {self.active_requests}/{self.max_concurrent})")
        else:
            logger.warning("请求超时，未能获取执行权限")

        return acquired

    def release(self):
        """释放执行权限"""
        if not self.enabled:
            return

        self.semaphore.release()

        with self.active_lock:
            self.active_requests -= 1
            self.completed_requests += 1

        logger.debug(f"请求释放执行权限 (活跃: {self.active_requests}/{self.max_concurrent})")

    def can_accept_request(self):
        """
        检查是否可以接受新请求

        Returns:
            tuple: (bool: 是否可接受, str: 拒绝原因)
        """
        if not self.enabled:
            return True, None

        with self.active_lock:
            # 如果有空闲槽位，直接接受
            if self.active_requests < self.max_concurrent:
                return True, None

            # 如果启用了队列且队列未满，可以排队
            if self.queue and not self.queue.full():
                return True, None

            # 否则拒绝
            if self.queue and self.queue.full():
                return False, self.config.QUEUE_FULL_MESSAGE
            else:
                return False, self.config.CONCURRENCY_LIMIT_MESSAGE

    def get_stats(self):
        """
        获取统计信息

        Returns:
            dict: 统计数据
        """
        with self.active_lock:
            return {
                "enabled": self.enabled,
                "max_concurrent": self.max_concurrent,
                "active_requests": self.active_requests,
                "max_queue_size": self.max_queue_size,
                "queue_size": self.queue.qsize() if self.queue else 0,
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "rejected_requests": self.rejected_requests,
            }

    def limit_concurrency(self, func):
        """
        装饰器：为函数添加并发限制

        Args:
            func: 要装饰的函数

        Returns:
            wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否可以接受请求
            with self.active_lock:
                self.total_requests += 1

            can_accept, reason = self.can_accept_request()

            if not can_accept:
                with self.active_lock:
                    self.rejected_requests += 1
                logger.warning(f"请求被拒绝: {reason} (总请求: {self.total_requests}, 已拒绝: {self.rejected_requests})")
                return jsonify({
                    "error": reason,
                    "retry_after": 5  # 建议5秒后重试
                }), 503  # Service Unavailable

            # 尝试获取执行权限
            acquired = self.acquire(timeout=self.request_timeout)

            if not acquired:
                with self.active_lock:
                    self.rejected_requests += 1
                logger.error("获取执行权限超时")
                return jsonify({
                    "error": "请求超时，请重试",
                    "retry_after": 3
                }), 504  # Gateway Timeout

            try:
                # 执行实际函数
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                logger.error(f"请求处理出错: {str(e)}", exc_info=True)
                raise

            finally:
                # 释放权限
                self.release()

        return wrapper

    def limit_streaming(self, generator_func):
        """
        装饰器：为流式函数添加并发限制

        Args:
            generator_func: 生成器函数

        Returns:
            wrapped generator function
        """
        @wraps(generator_func)
        def wrapper(*args, **kwargs):
            # 检查是否可以接受请求
            with self.active_lock:
                self.total_requests += 1

            can_accept, reason = self.can_accept_request()

            if not can_accept:
                with self.active_lock:
                    self.rejected_requests += 1
                logger.warning(f"流式请求被拒绝: {reason}")

                # 返回错误流
                def error_stream():
                    import json
                    error_data = {
                        "type": "error",
                        "message": reason
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

                return error_stream()

            # 尝试获取执行权限
            acquired = self.acquire(timeout=self.request_timeout)

            if not acquired:
                with self.active_lock:
                    self.rejected_requests += 1

                def timeout_stream():
                    import json
                    error_data = {
                        "type": "error",
                        "message": "请求超时，请重试"
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

                return timeout_stream()

            try:
                # 执行生成器函数
                for item in generator_func(*args, **kwargs):
                    yield item

            except Exception as e:
                logger.error(f"流式请求处理出错: {str(e)}", exc_info=True)
                raise

            finally:
                # 释放权限
                self.release()

        return wrapper


# 创建全局限制器实例（将在app.py中初始化）
limiter = None


def init_limiter(config):
    """
    初始化全局限制器

    Args:
        config: 配置对象
    """
    global limiter
    limiter = ConcurrencyLimiter(config)
    return limiter


def get_limiter():
    """
    获取全局限制器实例

    Returns:
        ConcurrencyLimiter: 限制器实例
    """
    return limiter

