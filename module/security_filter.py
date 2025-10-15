"""
对话安全过滤器模块
提供输入验证、内容过滤和安全防护功能
"""

import re
from typing import Tuple, Optional
from loguru import logger


class SecurityFilter:
    """对话安全过滤器"""

    def __init__(self, config):
        self.config = config
        self.max_input_length = config.MAX_INPUT_LENGTH
        self.enable_truncation = config.ENABLE_INPUT_TRUNCATION
        self.enable_security_filter = config.ENABLE_SECURITY_FILTER
        self.blocked_keywords = config.SECURITY_BLOCKED_MESSAGES
        self.security_response = config.SECURITY_RESPONSE_TEMPLATE

        logger.info(f"安全过滤器初始化完成:")
        logger.info(f"  - 输入长度限制: {self.max_input_length} 字符")
        logger.info(f"  - 截断模式: {'启用' if self.enable_truncation else '拒绝'}")
        logger.info(f"  - 安全过滤: {'启用' if self.enable_security_filter else '禁用'}")
        logger.info(f"  - 敏感关键词数量: {len(self.blocked_keywords)}")
        logger.debug(f"  - 敏感关键词列表: {self.blocked_keywords[:5]}...")

    def validate_input(self, user_input: str) -> Tuple[bool, str, Optional[str]]:
        """
        验证用户输入

        Args:
            user_input: 用户输入内容

        Returns:
            Tuple[bool, str, Optional[str]]: (是否通过验证, 处理后的输入, 错误信息)
        """
        if not user_input or not user_input.strip():
            return False, "", "输入内容不能为空"

        original_input = user_input
        processed_input = user_input.strip()

        # 1. 长度检查
        length_result = self._check_length(processed_input)
        if not length_result[0]:
            return length_result

        processed_input = length_result[1]

        # 2. 安全过滤
        if self.enable_security_filter:
            logger.debug(f"开始安全过滤检查: '{processed_input}'")
            logger.debug(f"敏感关键词列表: {self.blocked_keywords[:3]}...")
            security_result = self._check_security(processed_input)
            if not security_result[0]:
                logger.warning(f"安全过滤触发: 用户输入包含敏感内容")
                logger.debug(f"原始输入: {original_input[:100]}...")
                return security_result
            else:
                logger.debug("安全过滤检查通过")

        logger.debug(f"输入验证通过: {len(processed_input)} 字符")
        return True, processed_input, None

    def _check_length(self, user_input: str) -> Tuple[bool, str, Optional[str]]:
        """检查输入长度"""
        if len(user_input) <= self.max_input_length:
            return True, user_input, None

        if self.enable_truncation:
            # 截断输入
            truncated_input = user_input[:self.max_input_length]
            logger.info(f"输入过长，已截断: {len(user_input)} -> {len(truncated_input)} 字符")
            return True, truncated_input, None
        else:
            # 拒绝输入
            error_msg = f"输入内容过长，最大允许 {self.max_input_length} 字符，当前 {len(user_input)} 字符"
            logger.warning(f"输入被拒绝: {error_msg}")
            return False, "", error_msg

    def _check_security(self, user_input: str) -> Tuple[bool, str, Optional[str]]:
        """检查安全风险"""
        user_input_lower = user_input.lower()

        # 检查是否包含敏感关键词
        for keyword in self.blocked_keywords:
            if keyword.lower() in user_input_lower:
                logger.warning(f"检测到敏感关键词: '{keyword}' in '{user_input[:50]}...'")
                return False, "", self.security_response

        # 检查是否试图获取系统信息
        system_patterns = [
            r'你的.*提示词|your.*prompt',
            r'系统.*配置|system.*config',
            r'模型.*信息|model.*info',
            r'知识库.*内容|knowledge.*content',
            r'原始.*文档|raw.*document',
            r'向量.*数据|vector.*data',
            r'检索.*过程|retrieval.*process',
            r'embedding.*模型|嵌入.*模型'
        ]

        for pattern in system_patterns:
            if re.search(pattern, user_input_lower):
                logger.debug(f"检测到系统信息查询模式: {pattern}")
                return False, "", self.security_response

        # 检查是否包含过多技术术语（可能的攻击尝试）
        tech_terms = ['api', 'endpoint', 'database', 'server', 'backend', 'frontend',
                     'api', '接口', '数据库', '服务器', '后端', '前端']
        tech_count = sum(1 for term in tech_terms if term.lower() in user_input_lower)

        if tech_count >= 3:  # 如果包含3个或以上技术术语
            logger.debug(f"检测到过多技术术语: {tech_count} 个")
            return False, "", self.security_response

        return True, user_input, None

    def get_security_stats(self) -> dict:
        """获取安全统计信息"""
        return {
            "enabled": self.enable_security_filter,
            "max_input_length": self.max_input_length,
            "truncation_enabled": self.enable_truncation,
            "blocked_keywords_count": len(self.blocked_keywords),
            "blocked_keywords": self.blocked_keywords[:5] if self.blocked_keywords else []  # 只显示前5个
        }
