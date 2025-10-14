"""
查询扩展模块
负责将用户问题扩展为更适合检索的关键词
"""

import re
import logging

logger = logging.getLogger(__name__)


class QueryExpander:
    """查询扩展器"""

    def __init__(self, llm, config):
        """
        初始化查询扩展器

        Args:
            llm: 语言模型
            config: 配置对象
        """
        self.llm = llm
        self.config = config
        self.enabled = config.ENABLE_QUERY_EXPANSION

    def expand(self, question):
        """
        扩展用户问题为检索关键词

        Args:
            question: 用户原始问题

        Returns:
            expanded_query: 扩展后的查询（如果失败则返回原问题）
        """
        if not self.enabled:
            logger.debug("查询扩展已禁用")
            return question

        try:
            logger.debug(f"查询扩展: {question}")

            # 使用模板生成扩展查询
            expansion_prompt = self.config.QUERY_EXPANSION_TEMPLATE.format(
                question=question
            )
            expanded_query = self.llm.invoke(expansion_prompt)

            if not expanded_query or len(expanded_query.strip()) == 0:
                logger.debug("扩展查询为空，使用原查询")
                return question

            # 清理扩展结果
            cleaned_query = self._clean_query(expanded_query)

            if len(cleaned_query) > 0:
                logger.info(f"查询扩展: {question} → {cleaned_query}")
                return cleaned_query
            else:
                logger.debug("清理后查询为空，使用原查询")
                return question

        except Exception as e:
            logger.warning(f"查询扩展失败: {e}")
            return question

    def _clean_query(self, query):
        """
        清理扩展后的查询

        Args:
            query: 原始扩展查询

        Returns:
            cleaned_query: 清理后的查询
        """
        # 基本清理
        cleaned = query.strip()

        # 移除序号（1. 2) 等格式）
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned, flags=re.MULTILINE)

        # 换行转空格
        cleaned = re.sub(r'\n+', ' ', cleaned)

        # 多空格转单空格
        cleaned = re.sub(r'\s+', ' ', cleaned)

        cleaned = cleaned.strip()

        # 限制长度
        max_length = 50
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            logger.debug(f"查询过长，截断到{max_length}字符")

        return cleaned

