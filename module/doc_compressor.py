#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档压缩模块 - 混合方案（TextRank + m3e重排序）
"""

import re
import logging
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DocumentCompressor:
    """
    文档压缩器

    使用混合方案压缩检索到的文档：
    1. TextRank算法初筛（快速过滤无关句子）
    2. m3e语义重排序（精确选择最相关句子）
    """

    def __init__(self, embeddings, config):
        """
        初始化文档压缩器

        Args:
            embeddings: HuggingFace embeddings实例（m3e模型）
            config: 配置对象
        """
        self.embeddings = embeddings
        self.config = config
        self.enable_compression = getattr(config, 'ENABLE_DOC_COMPRESSION', True)
        self.max_sentences = getattr(config, 'MAX_SENTENCES_PER_DOC', 3)
        self.min_sentence_length = getattr(config, 'MIN_SENTENCE_LENGTH', 5)

        # 尝试导入jieba（用于TextRank）
        self.jieba_available = False
        try:
            import jieba
            import jieba.analyse
            self.jieba = jieba
            self.jieba_available = True
            logger.info("jieba分词库加载成功，将使用TextRank初筛")
        except ImportError:
            logger.warning("jieba未安装，将直接使用m3e重排序（推荐安装: pip install jieba）")

        logger.info("文档压缩器初始化完成")
        logger.info(f"  - 压缩功能: {'启用' if self.enable_compression else '禁用'}")
        logger.info(f"  - 每文档保留: {self.max_sentences} 个句子")
        logger.info(f"  - TextRank初筛: {'可用' if self.jieba_available else '不可用'}")

    def compress_documents(self, documents, question: str) -> List:
        """
        压缩文档列表

        Args:
            documents: 检索到的文档列表
            question: 用户问题

        Returns:
            List: 压缩后的文档列表
        """
        if not self.enable_compression:
            logger.debug("文档压缩未启用，返回原始文档")
            return documents

        if not documents:
            return documents

        logger.info(f"开始压缩 {len(documents)} 个文档...")
        compressed_docs = []

        for i, doc in enumerate(documents, 1):
            original_text = doc.page_content
            original_length = len(original_text)

            # 压缩文档
            compressed_text = self._compress_single_doc(original_text, question)
            compressed_length = len(compressed_text)

            # 计算压缩率
            compression_ratio = (1 - compressed_length / original_length) * 100 if original_length > 0 else 0

            logger.debug(f"文档 {i}: {original_length}字 → {compressed_length}字 (压缩 {compression_ratio:.1f}%)")

            # 保留元数据，更新内容
            compressed_doc = type(doc)(
                page_content=compressed_text,
                metadata=doc.metadata
            )
            compressed_docs.append(compressed_doc)

        total_original = sum(len(d.page_content) for d in documents)
        total_compressed = sum(len(d.page_content) for d in compressed_docs)
        total_ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0

        logger.info(f"文档压缩完成: {total_original}字 → {total_compressed}字 (总压缩率: {total_ratio:.1f}%)")

        return compressed_docs

    def _compress_single_doc(self, text: str, question: str) -> str:
        """
        压缩单个文档

        Args:
            text: 文档文本
            question: 用户问题

        Returns:
            str: 压缩后的文本
        """
        # 分句
        sentences = self._split_sentences(text)

        if len(sentences) <= self.max_sentences:
            # 文档本身就很短，无需压缩
            return text

        # 方案1：如果有jieba，先用TextRank初筛
        if self.jieba_available and len(sentences) > 10:
            sentences = self._textrank_filter(sentences, top_k=min(10, len(sentences)))

        # 方案2：使用m3e语义重排序
        if len(sentences) > self.max_sentences:
            sentences = self._m3e_rerank(sentences, question, top_k=self.max_sentences)

        # 拼接句子
        compressed_text = '。'.join(sentences)
        if compressed_text and not compressed_text.endswith('。'):
            compressed_text += '。'

        return compressed_text

    def _split_sentences(self, text: str) -> List[str]:
        """
        分句

        Args:
            text: 文本

        Returns:
            List[str]: 句子列表
        """
        # 按句号、问号、感叹号分句
        sentences = re.split(r'[。！？\n]+', text)

        # 过滤太短的句子
        sentences = [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]

        return sentences

    def _textrank_filter(self, sentences: List[str], top_k: int = 10) -> List[str]:
        """
        使用TextRank算法初筛句子

        Args:
            sentences: 句子列表
            top_k: 保留句子数量

        Returns:
            List[str]: 初筛后的句子
        """
        if not self.jieba_available or len(sentences) <= top_k:
            return sentences

        try:
            # 将所有句子合并
            full_text = '。'.join(sentences)

            # 使用TextRank提取关键词
            keywords = self.jieba.analyse.textrank(full_text, topK=20, withWeight=False)

            # 对每个句子打分（包含的关键词数量）
            scored_sentences = []
            for sent in sentences:
                score = sum(1 for kw in keywords if kw in sent)
                scored_sentences.append((score, sent))

            # 排序并取top-k
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            filtered = [sent for _, sent in scored_sentences[:top_k]]

            logger.debug(f"TextRank初筛: {len(sentences)}句 → {len(filtered)}句")
            return filtered

        except Exception as e:
            logger.warning(f"TextRank初筛失败: {e}，使用原始句子")
            return sentences

    def _m3e_rerank(self, sentences: List[str], question: str, top_k: int = 3) -> List[str]:
        """
        使用m3e模型进行语义重排序

        Args:
            sentences: 句子列表
            question: 用户问题
            top_k: 保留句子数量

        Returns:
            List[str]: 重排序后的top-k句子
        """
        if len(sentences) <= top_k:
            return sentences

        try:
            logger.debug(f"m3e重排序: 对 {len(sentences)} 个句子进行语义相似度计算")

            # 1. 获取问题的embedding
            question_emb = self.embeddings.embed_query(question)
            question_emb = np.array(question_emb).reshape(1, -1)

            # 2. 获取所有句子的embeddings
            sentence_embs = self.embeddings.embed_documents(sentences)
            sentence_embs = np.array(sentence_embs)

            # 3. 计算余弦相似度
            similarities = cosine_similarity(question_emb, sentence_embs)[0]

            # 4. 排序并取top-k
            top_indices = similarities.argsort()[-top_k:][::-1]  # 降序

            # 5. 按原文顺序返回（保持上下文连贯性）
            top_indices_sorted = sorted(top_indices)
            reranked = [sentences[i] for i in top_indices_sorted]

            # 记录相似度信息（debug）
            if logger.isEnabledFor(logging.DEBUG):
                for idx in top_indices:
                    logger.debug(f"  - 句子[{idx}]: 相似度={similarities[idx]:.3f}, 内容=\"{sentences[idx][:30]}...\"")

            logger.debug(f"m3e重排序完成: 选出 {len(reranked)} 个最相关句子")
            return reranked

        except Exception as e:
            logger.error(f"m3e重排序失败: {e}")
            # 降级：返回前top_k个句子
            return sentences[:top_k]

    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取关键词（用于降级方案）

        Args:
            text: 文本
            top_k: 提取关键词数量

        Returns:
            List[str]: 关键词列表
        """
        if self.jieba_available:
            # 使用jieba的TF-IDF或TextRank
            keywords = self.jieba.analyse.extract_tags(text, topK=top_k)
            return keywords
        else:
            # 简单的n-gram提取（移除标点和空白）
            # 移除标点符号、空格、制表符、换行符
            text = re.sub(r'[，。！？、；：""''（）【】《》 \t\n\r]+', '', text)
            keywords = []
            # 2-3字的词
            for i in range(len(text) - 1):
                keywords.append(text[i:i+2])
            for i in range(len(text) - 2):
                keywords.append(text[i:i+3])
            # 去重并返回
            return list(set(keywords))[:top_k]

    def get_stats(self) -> dict:
        """
        获取压缩器统计信息

        Returns:
            dict: 统计信息
        """
        return {
            "enabled": self.enable_compression,
            "max_sentences": self.max_sentences,
            "min_sentence_length": self.min_sentence_length,
            "textrank_available": self.jieba_available
        }

