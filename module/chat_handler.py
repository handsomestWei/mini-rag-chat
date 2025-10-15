"""
对话处理模块
负责处理用户查询和生成回答
"""

import time
import logging
import psutil
import os
import json
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调处理器"""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当LLM生成新token时调用"""
        self.tokens.append(token)

    def get_text(self):
        """获取完整文本"""
        return "".join(self.tokens)

    def clear(self):
        """清空tokens"""
        self.tokens = []


class ChatHandler:
    """对话处理器"""

    def __init__(self, chain, query_expander, retriever, config, doc_compressor=None):
        """
        初始化对话处理器

        Args:
            chain: LangChain对话链
            query_expander: 查询扩展器
            retriever: 检索器
            config: 配置对象
            doc_compressor: 文档压缩器（可选）
        """
        self.chain = chain
        self.query_expander = query_expander
        self.retriever = retriever
        self.config = config
        self.doc_compressor = doc_compressor

        # 初始化安全过滤器
        try:
            from .security_filter import SecurityFilter
            self.security_filter = SecurityFilter(config)
            logger.info("安全过滤器初始化成功")
        except ImportError as e:
            logger.warning(f"安全过滤器初始化失败: {e}")
            self.security_filter = None

        # 初始化意图识别器
        self.intent_classifier = None
        if getattr(config, 'ENABLE_INTENT_CLASSIFICATION', False):
            try:
                from .intent_classifier import IntentClassifier
                self.intent_classifier = IntentClassifier(config)
                logger.info("意图识别器初始化成功")
            except ImportError as e:
                logger.warning(f"意图识别器初始化失败: {e}")
            except Exception as e:
                logger.warning(f"意图识别器加载失败: {e}")

    def _get_memory_usage(self):
        """获取当前进程内存使用情况"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        mem_percent = process.memory_percent()
        return mem_mb, mem_percent

    def handle_query(self, user_input):
        """
        处理用户查询

        Args:
            user_input: 用户输入的问题

        Returns:
            response: 包含回答、检索文档、性能统计的字典
        """
        logger.info("=" * 80)
        logger.info(f"收到用户问题: {user_input}")
        logger.info("=" * 80)

        # 安全验证
        if self.security_filter:
            is_valid, processed_input, error_msg = self.security_filter.validate_input(user_input)
            if not is_valid:
                logger.warning(f"输入验证失败: {error_msg}")
                return {
                    "answer": error_msg,
                    "source_documents": [],
                    "stats": {
                        "total_time": 0,
                        "retrieval_time": 0,
                        "llm_time": 0,
                        "doc_count": 0,
                        "memory_usage": self._get_memory_usage()[0]
                    }
                }
            user_input = processed_input
            logger.debug(f"输入验证通过，处理后长度: {len(user_input)} 字符")

        # 记录开始时间和内存
        start_time = time.time()
        mem_before, _ = self._get_memory_usage()

        try:
            # 第1步：查询扩展
            expanded_query = self.query_expander.expand(user_input)
            logger.debug(f"🔍 原始查询: {user_input}")
            logger.debug(f"🔍 扩展查询: {expanded_query}")

            # 第2步：文档检索
            logger.info("步骤1: 文档检索")
            retrieval_start = time.time()

            relevant_docs = self.retriever.get_relevant_documents(expanded_query)

            retrieval_time = time.time() - retrieval_start
            logger.info(f"检索完成 (耗时: {retrieval_time:.3f}秒)")
            logger.info(f"检索到 {len(relevant_docs)} 个相关文档")

            # 记录检索到的文档（debug级别，使用单条日志避免重复）
            doc_log_lines = ["=" * 80, "📄 检索到的文档详情:", "=" * 80]
            for i, doc in enumerate(relevant_docs, 1):
                doc_log_lines.append(f"\n【文档 {i}/{len(relevant_docs)}】")

                # 打印元数据
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_log_lines.append(f"📋 元数据: {doc.metadata}")

                # 打印文档内容（完整内容）
                content = doc.page_content
                doc_log_lines.append(f"📝 内容长度: {len(content)} 字符")
                doc_log_lines.append(f"💬 完整内容:\n{content}")
                doc_log_lines.append("-" * 60)

            doc_log_lines.append("=" * 80)
            logger.debug("\n".join(doc_log_lines))

            # 第3步：LLM生成回答
            logger.info("步骤2: LLM生成回答")
            llm_start = time.time()

            # 使用原始用户问题
            chinese_question = user_input

            # Debug: 打印发送给LLM的完整上下文（使用单条日志避免重复）
            context_log = (
                "=" * 80 + "\n" +
                "💭 发送给LLM的上下文:\n" +
                "=" * 80 + "\n" +
                f"📋 问题: {chinese_question}\n" +
                f"📚 检索到的文档数量: {len(relevant_docs)}\n" +
                "=" * 80
            )
            logger.debug(context_log)

            # 调用chain生成回答
            result = self.chain({"question": chinese_question, "chat_history": []})

            llm_time = time.time() - llm_start
            logger.info(f"LLM回答完成 (耗时: {llm_time:.3f}秒)")

            # 获取回答
            answer = result["answer"]
            answer_preview = answer[:150].replace('\n', ' ')
            logger.info(f"生成的回答 (预览): {answer_preview}...")

            # Debug: 打印完整的LLM回答（使用单条日志避免重复）
            answer_log = (
                "=" * 80 + "\n" +
                "💬 LLM生成的完整回答:\n" +
                "=" * 80 + "\n" +
                f"📝 回答长度: {len(answer)} 字符\n" +
                f"💭 完整内容:\n{answer}\n" +
                "=" * 80
            )
            logger.debug(answer_log)

            # 记录总耗时和内存
            total_time = time.time() - start_time
            mem_after, mem_percent = self._get_memory_usage()

            logger.info("性能统计:")
            logger.info(f"  - 检索耗时: {retrieval_time:.3f}秒 ({retrieval_time/total_time*100:.1f}%)")
            logger.info(f"  - LLM耗时: {llm_time:.3f}秒 ({llm_time/total_time*100:.1f}%)")
            logger.info(f"  - 总耗时: {total_time:.3f}秒")
            logger.debug(f"  - 内存变化: {mem_before:.2f}MB → {mem_after:.2f}MB")
            logger.info("=" * 80)

            # 返回结果
            return {
                "answer": answer,
                "relevant_docs": relevant_docs,
                "stats": {
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "doc_count": len(relevant_docs)
                }
            }

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"处理请求时出错: {str(e)}")
            logger.error("=" * 80)
            logger.error("详细错误信息:", exc_info=True)
            raise

    def handle_query_stream(self, user_input):
        """
        处理用户查询（流式输出）

        Args:
            user_input: 用户输入的问题

        Yields:
            dict: 流式数据块，包含 token 或状态信息
        """
        logger.info("=" * 80)
        logger.info(f"收到用户问题（流式）: {user_input}")
        logger.info("=" * 80)

        # 安全验证
        if self.security_filter:
            is_valid, processed_input, error_msg = self.security_filter.validate_input(user_input)
            if not is_valid:
                logger.warning(f"输入验证失败: {error_msg}")
                yield {
                    "type": "error",
                    "message": error_msg
                }
                return
            user_input = processed_input
            logger.debug(f"输入验证通过，处理后长度: {len(user_input)} 字符")

        start_time = time.time()

        try:
            # 意图识别
            intent_result = None
            if self.intent_classifier:
                intent_result = self.intent_classifier.classify(user_input)
                logger.info(f"🎯 意图识别: {intent_result['intent']} "
                           f"(置信度: {intent_result['confidence']:.3f}, "
                           f"方法: {intent_result['method']})")

                # 如果可以跳过RAG，直接返回预定义回复
                if intent_result['skip_rag'] and intent_result['response']:
                    logger.info(f"跳过RAG，直接返回预定义回复")
                    total_time = time.time() - start_time

                    # 发送完整回复
                    yield {
                        "type": "token",
                        "content": intent_result['response']
                    }

                    # 发送完成信号
                    yield {
                        "type": "done",
                        "stats": {
                            "intent": intent_result['intent'],
                            "method": intent_result['method'],
                            "confidence": intent_result['confidence'],
                            "skip_rag": True,
                            "total_time": total_time
                        }
                    }
                    return

            # 第1步：发送检索状态
            yield {
                "type": "status",
                "message": self.config.STREAM_STATUS_RETRIEVING,
                "stage": "retrieval"
            }

            # 查询扩展
            expanded_query = self.query_expander.expand(user_input)
            logger.debug(f"🔍 原始查询: {user_input}")
            logger.debug(f"🔍 扩展查询: {expanded_query}")

            # 文档检索
            logger.info("步骤1: 文档检索")
            retrieval_start = time.time()
            relevant_docs = self.retriever.get_relevant_documents(expanded_query)
            retrieval_time = time.time() - retrieval_start

            logger.info(f"检索完成 (耗时: {retrieval_time:.3f}秒)")
            logger.info(f"检索到 {len(relevant_docs)} 个相关文档")

            # 记录检索到的原始文档（debug级别，使用单条日志避免重复）
            doc_log_lines = ["=" * 80, "📄 检索到的原始文档:", "=" * 80]
            for i, doc in enumerate(relevant_docs, 1):
                doc_log_lines.append(f"\n【原始文档 {i}/{len(relevant_docs)}】")
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_log_lines.append(f"📋 元数据: {doc.metadata}")
                doc_log_lines.append(f"📝 长度: {len(doc.page_content)} 字符")
                doc_log_lines.append(f"💬 内容:\n{doc.page_content}")
                doc_log_lines.append("-" * 60)
            doc_log_lines.append("=" * 80)
            logger.debug("\n".join(doc_log_lines))

            # 文档压缩
            if self.doc_compressor:
                compression_start = time.time()
                relevant_docs = self.doc_compressor.compress_documents(relevant_docs, user_input)
                compression_time = time.time() - compression_start
                logger.info(f"文档压缩完成 (耗时: {compression_time:.3f}秒)")

                # 记录压缩后的文档（debug级别，使用单条日志避免重复）
                compressed_log_lines = ["=" * 80, "📄 压缩后的文档:", "=" * 80]
                for i, doc in enumerate(relevant_docs, 1):
                    compressed_log_lines.append(f"\n【压缩文档 {i}/{len(relevant_docs)}】")
                    compressed_log_lines.append(f"📝 长度: {len(doc.page_content)} 字符")
                    compressed_log_lines.append(f"💬 内容:\n{doc.page_content}")
                    compressed_log_lines.append("-" * 60)
                compressed_log_lines.append("=" * 80)
                logger.debug("\n".join(compressed_log_lines))
            else:
                compression_time = 0
                logger.debug("文档压缩器未启用")

            # 第2步：发送生成状态
            yield {
                "type": "status",
                "message": self.config.STREAM_STATUS_GENERATING,
                "stage": "generation"
            }

            # 第3步：准备上下文
            logger.info("步骤2: LLM流式生成回答")
            llm_start = time.time()

            # 手动构建上下文（因为流式需要直接调用LLM）
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # 构建完整提示词
            from langchain.prompts import PromptTemplate
            prompt_template = PromptTemplate(
                template=self.config.SYSTEM_PROMPT + "\n\n" + self.config.USER_QUESTION_TEMPLATE,
                input_variables=["context", "question"]
            )

            full_prompt = prompt_template.format(
                context=context,
                question=user_input
            )

            # Debug: 打印完整的提示词（使用单条日志避免重复）
            prompt_log = (
                "=" * 80 + "\n" +
                "💭 发送给LLM的完整提示词（流式模式）:\n" +
                "=" * 80 + "\n" +
                full_prompt + "\n" +
                "=" * 80
            )
            logger.debug(prompt_log)

            # 使用 LLM 的流式接口
            # 需要访问原始 LLM 对象
            llm = self.chain.combine_docs_chain.llm_chain.llm

            # 流式生成
            full_answer = ""
            for chunk in llm.stream(full_prompt):
                if chunk:
                    full_answer += chunk
                    yield {
                        "type": "token",
                        "content": chunk
                    }

            llm_time = time.time() - llm_start
            total_time = time.time() - start_time

            logger.info(f"LLM流式回答完成 (耗时: {llm_time:.3f}秒)")
            logger.info(f"总耗时: {total_time:.3f}秒")

            # Debug: 打印流式生成的完整回答（使用单条日志避免重复）
            answer_log = (
                "=" * 80 + "\n" +
                "💬 LLM流式生成的完整回答:\n" +
                "=" * 80 + "\n" +
                f"📝 回答长度: {len(full_answer)} 字符\n" +
                f"💭 完整内容:\n{full_answer}\n" +
                "=" * 80
            )
            logger.debug(answer_log)

            # 第4步：发送完成状态
            stats = {
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": total_time,
                "doc_count": len(relevant_docs),
                "skip_rag": False
            }

            # 添加压缩时间信息
            if self.doc_compressor and compression_time > 0:
                stats["compression_time"] = compression_time

            # 添加意图识别信息
            if intent_result:
                stats.update({
                    "intent": intent_result['intent'],
                    "intent_method": intent_result['method'],
                    "intent_confidence": intent_result['confidence']
                })

            yield {
                "type": "done",
                "stats": stats
            }

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"流式处理请求时出错: {str(e)}")
            logger.error("=" * 80)
            logger.error("详细错误信息:", exc_info=True)

            yield {
                "type": "error",
                "message": f"处理出错: {str(e)}"
            }

