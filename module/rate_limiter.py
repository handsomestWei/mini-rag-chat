#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接口频率限制模块
结合 Flask-Limiter 和自定义脚本检测
"""

import logging
from flask import request, jsonify
from functools import wraps

logger = logging.getLogger(__name__)


def get_real_ip():
    """
    获取客户端真实IP地址

    考虑代理服务器的情况：
    1. X-Forwarded-For: 代理链中的原始客户端IP
    2. X-Real-IP: Nginx等设置的真实IP
    3. remote_addr: 直接连接的IP

    Returns:
        str: 客户端IP地址
    """
    # 如果配置信任代理头
    from config import TRUST_PROXY_HEADERS

    if TRUST_PROXY_HEADERS:
        # 1. 优先从 X-Forwarded-For 获取（可能包含多个IP，取第一个）
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # X-Forwarded-For 格式: "client, proxy1, proxy2"
            # 取第一个IP（客户端真实IP）
            ip = forwarded_for.split(',')[0].strip()
            logger.debug(f"从 X-Forwarded-For 获取IP: {ip}")
            return ip

        # 2. 其次从 X-Real-IP 获取
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            logger.debug(f"从 X-Real-IP 获取IP: {real_ip}")
            return real_ip

    # 3. 最后使用直接连接的IP
    ip = request.remote_addr
    logger.debug(f"使用 remote_addr: {ip}")
    return ip


def check_bot_detection():
    """
    检测可疑的脚本/机器人行为

    Returns:
        tuple: (is_suspicious, reason)
    """
    from config import (
        ENABLE_BOT_DETECTION,
        SUSPICIOUS_USER_AGENTS,
        WHITELISTED_USER_AGENTS,
        BLOCK_EMPTY_USER_AGENT,
        CHECK_REFERER
    )

    if not ENABLE_BOT_DETECTION:
        return False, None

    user_agent = request.headers.get('User-Agent', '').lower()

    # 1. 检查空User-Agent
    if BLOCK_EMPTY_USER_AGENT and not user_agent:
        logger.warning(f"检测到空User-Agent请求: {request.remote_addr} - {request.path}")
        return True, "缺少User-Agent"

    # 2. 检查是否在白名单中（搜索引擎爬虫等）
    for whitelist in WHITELISTED_USER_AGENTS:
        if whitelist.lower() in user_agent:
            logger.debug(f"白名单User-Agent: {user_agent}")
            return False, None

    # 3. 检查可疑User-Agent
    for suspicious in SUSPICIOUS_USER_AGENTS:
        if suspicious.lower() in user_agent:
            logger.warning(f"检测到可疑User-Agent: {user_agent} from {request.remote_addr}")
            return True, f"可疑的User-Agent"

    # 4. 可选：检查Referer（可能误伤，谨慎启用）
    if CHECK_REFERER and request.path != '/' and request.method == 'POST':
        referer = request.headers.get('Referer', '')
        if not referer:
            logger.warning(f"POST请求缺少Referer: {request.remote_addr} - {request.path}")
            return True, "缺少Referer"

    return False, None


def bot_detection_required(f):
    """
    脚本检测装饰器
    在路由函数执行前检测是否为脚本请求
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_suspicious, reason = check_bot_detection()
        if is_suspicious:
            logger.warning(f"拦截可疑请求: {get_real_ip()} - {reason}")
            return jsonify({
                "error": "请求被拒绝",
                "status": "forbidden"
            }), 403
        return f(*args, **kwargs)
    return decorated_function


def init_rate_limiter(app, config_module):
    """
    初始化频率限制器

    Args:
        app: Flask应用实例
        config_module: 配置模块

    Returns:
        limiter: Flask-Limiter实例（如果启用）
    """
    if not config_module.ENABLE_RATE_LIMIT:
        logger.info("频率限制未启用")
        return None

    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        # 配置存储后端
        storage_uri = None
        if config_module.RATE_LIMIT_STORAGE == "redis":
            storage_uri = config_module.RATE_LIMIT_REDIS_URL
            logger.info(f"使用Redis存储: {storage_uri}")
        elif config_module.RATE_LIMIT_STORAGE == "memory":
            storage_uri = "memory://"
            logger.info("使用内存存储")

        # 创建限制器（使用自定义的get_real_ip）
        limiter = Limiter(
            key_func=get_real_ip,  # 使用自定义IP获取函数
            app=app,
            default_limits=[config_module.RATE_LIMIT_DEFAULT] if config_module.RATE_LIMIT_DEFAULT else [],
            storage_uri=storage_uri,
            strategy=config_module.RATE_LIMIT_STRATEGY,
            headers_enabled=config_module.RATE_LIMIT_HEADERS_ENABLED,
        )

        # 自定义错误处理
        @app.errorhandler(429)
        def ratelimit_handler(e):
            """处理频率限制错误"""
            ip = get_real_ip()
            user_agent = request.headers.get('User-Agent', 'Unknown')
            logger.warning(f"频率限制触发: IP={ip}, UA={user_agent[:50]}, Path={request.path}")

            return jsonify({
                "error": config_module.RATE_LIMIT_MESSAGE,
                "status": "rate_limit_exceeded"
            }), 429

        # 定义共享限制（用于路由装饰器）
        # 将列表转换为逗号分隔的字符串
        chat_limits = config_module.RATE_LIMIT_CHAT
        if isinstance(chat_limits, list):
            limiter.chat_limit = ";".join(chat_limits)
        else:
            limiter.chat_limit = chat_limits

        limiter.health_limit = config_module.RATE_LIMIT_HEALTH
        limiter.stats_limit = config_module.RATE_LIMIT_STATS

        logger.info(f"频率限制已启用: {config_module.RATE_LIMIT_STORAGE} 存储, 策略: {config_module.RATE_LIMIT_STRATEGY}")
        logger.info(f"默认限制: {config_module.RATE_LIMIT_DEFAULT}")
        logger.info(f"对话限制: {limiter.chat_limit}")
        logger.info(f"信任代理头: {config_module.TRUST_PROXY_HEADERS}")
        logger.info(f"脚本检测: {'启用' if config_module.ENABLE_BOT_DETECTION else '禁用'}")

        return limiter

    except ImportError as e:
        logger.error(f"无法导入flask-limiter: {e}")
        logger.error("请运行: pip install flask-limiter")
        return None
    except Exception as e:
        logger.error(f"初始化频率限制器失败: {e}")
        return None


# 创建一个全局变量来存储limiter实例
_limiter_instance = None

def set_limiter(limiter):
    """设置全局limiter实例"""
    global _limiter_instance
    _limiter_instance = limiter

def get_limiter():
    """获取全局limiter实例"""
    return _limiter_instance


def get_client_info():
    """
    获取客户端详细信息（用于日志和调试）

    Returns:
        dict: 客户端信息
    """
    return {
        "ip": get_real_ip(),
        "remote_addr": request.remote_addr,
        "x_forwarded_for": request.headers.get('X-Forwarded-For'),
        "x_real_ip": request.headers.get('X-Real-IP'),
        "user_agent": request.headers.get('User-Agent', ''),
        "referer": request.headers.get('Referer', ''),
        "path": request.path,
        "method": request.method
    }

