"""
Mini RAG Chat - 知识问答系统
"""

import os
import logging
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# 导入配置和模块
import config
from module.rag_manager import RAGManager
from module.query_expander import QueryExpander
from module.chat_handler import ChatHandler
from module.concurrency_limiter import init_limiter, get_limiter
from module.rate_limiter import init_rate_limiter, bot_detection_required, get_client_info

# 配置日志（添加线程信息）
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - [%(threadName)s-%(thread)d] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)


def log_memory():
    """记录内存使用情况"""
    if config.ENABLE_MEMORY_MONITOR:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        mem_percent = process.memory_percent()
        logger.info(f"内存使用: {mem_mb:.2f}MB ({mem_percent:.1f}%)")
        if mem_percent > config.MEMORY_WARNING_THRESHOLD:
            logger.warning(f"⚠️  内存使用率过高: {mem_percent:.1f}%")


def initialize_system():
    """初始化系统组件"""
    logger.info("=" * 50)
    logger.info("启动 Mini RAG Chat")
    logger.info(f"配置: {config.EMBEDDING_DEVICE.upper()}, Batch={config.EMBEDDING_BATCH_SIZE}")
    log_memory()

    # 1. 加载嵌入模型
    logger.info("正在加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={
            "batch_size": config.EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": True
        }
    )
    logger.info("嵌入模型加载完成")
    log_memory()

    # 2. 初始化RAG管理器并加载向量库
    logger.info("初始化RAG管理器...")
    rag_manager = RAGManager(config, embeddings)
    vector_store = rag_manager.load_vector_store()
    logger.info("向量库准备完成")
    log_memory()

    # 3. 加载LLM模型
    logger.info(f"正在加载 LLM 模型: {config.LLM_MODEL}")
    logger.info(f"性能参数: ctx={config.OLLAMA_NUM_CTX}, threads={config.OLLAMA_NUM_THREAD}")

    llm = Ollama(
        model=config.LLM_MODEL,
        num_ctx=config.OLLAMA_NUM_CTX,
        num_thread=config.OLLAMA_NUM_THREAD,
        num_predict=config.OLLAMA_NUM_PREDICT,
        temperature=config.OLLAMA_TEMPERATURE,
        top_k=config.OLLAMA_TOP_K,
        top_p=config.OLLAMA_TOP_P,
        repeat_penalty=config.OLLAMA_REPEAT_PENALTY
    )
    logger.info("LLM 模型加载完成")

    # 4. 创建检索器
    retriever_kwargs = {
        "k": config.RETRIEVER_K,
        "fetch_k": config.RETRIEVER_FETCH_K
    }

    # 根据检索模式配置参数
    search_type = getattr(config, 'SEARCH_TYPE', 'similarity')
    if search_type == "mmr":
        retriever_kwargs["lambda_mult"] = config.MMR_DIVERSITY_SCORE
        logger.info(f"使用MMR检索模式，多样性: {config.MMR_DIVERSITY_SCORE}")
    elif search_type == "similarity_score_threshold":
        retriever_kwargs["score_threshold"] = config.SCORE_THRESHOLD
        logger.info(f"使用阈值检索模式，阈值: {config.SCORE_THRESHOLD}")
    else:
        logger.info(f"使用{search_type}检索模式")

    retriever = rag_manager.get_retriever(
        search_type=search_type,
        search_kwargs=retriever_kwargs
    )

    # 5. 创建对话链
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 组合提示词
    combine_prompt = PromptTemplate(
        template=config.SYSTEM_PROMPT + "\n\n" + config.USER_QUESTION_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        verbose=False,
        return_source_documents=False
    )
    logger.info("RAG链初始化完成")

    # 6. 创建查询扩展器
    query_expander = QueryExpander(llm, config)

    # 7. 创建文档压缩器
    from module.doc_compressor import DocumentCompressor
    doc_compressor = DocumentCompressor(embeddings, config)

    # 8. 创建对话处理器
    chat_handler = ChatHandler(chain, query_expander, retriever, config, doc_compressor)

    # 9. 初始化并发限制器
    limiter = init_limiter(config)
    if config.ENABLE_CONCURRENCY_LIMIT:
        logger.info(f"并发限制已启用: 最大并发={config.MAX_CONCURRENT_REQUESTS}, 队列={config.MAX_QUEUE_SIZE}")
    else:
        logger.info("并发限制已禁用")

    log_memory()
    logger.info("=" * 50)

    return {
        "rag_manager": rag_manager,
        "chat_handler": chat_handler,
        "llm": llm,
        "embeddings": embeddings,
        "limiter": limiter
    }


# 初始化系统组件（全局）
components = initialize_system()

# 初始化频率限制器（必须在路由定义之前）
rate_limiter = init_rate_limiter(app, config)
components["rate_limiter"] = rate_limiter


# ========== Flask路由 ==========

@app.route("/")
def index():
    """首页"""
    # 传递配置到前端
    web_config = {
        "appTitle": config.WEB_APP_TITLE,
        "appSubtitle": config.WEB_APP_SUBTITLE,
        "headerIcon": config.WEB_HEADER_ICON,
        "userIcon": config.WEB_USER_ICON,
        "aiIcon": config.WEB_AI_ICON,
        "welcomeTitle": config.WEB_WELCOME_TITLE,
        "welcomeMessage": config.WEB_WELCOME_MESSAGE,
        "welcomeHint": config.WEB_WELCOME_HINT,
        "primaryColor": config.WEB_PRIMARY_COLOR,
        "secondaryColor": config.WEB_SECONDARY_COLOR,
        "accentColor": config.WEB_ACCENT_COLOR,
        "accentSecondary": config.WEB_ACCENT_SECONDARY,
        "enableStreaming": config.WEB_ENABLE_STREAMING,
        "errorNoResponse": config.ERROR_NO_RESPONSE,
    }
    return render_template("index.html", config=web_config)


@app.route("/chat", methods=["POST"])
def chat():
    """处理用户问答请求（非流式，兼容旧版）"""
    # 应用并发限制
    limiter = components["limiter"]

    @limiter.limit_concurrency
    def _handle_chat():
        try:
            user_input = request.form["user_input"]

            # 使用对话处理器处理查询
            response = components["chat_handler"].handle_query(user_input)

            return response["answer"]

        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}"

    return _handle_chat()


@app.route("/chat/stream", methods=["POST"])
@bot_detection_required  # 脚本检测
def chat_stream():
    """处理用户问答请求（流式输出）"""
    try:
        user_input = request.form.get("user_input") or request.json.get("user_input")

        if not user_input:
            return jsonify({"error": "缺少user_input参数"}), 400

        # 记录客户端信息（debug模式）
        if config.LOG_LEVEL == "DEBUG":
            client_info = get_client_info()
            logger.debug(f"客户端信息: IP={client_info['ip']}, UA={client_info['user_agent'][:50]}")

        # 应用并发限制到生成器
        limiter = components["limiter"]

        @limiter.limit_streaming
        def generate():
            """生成器函数，用于流式输出"""
            try:
                for chunk in components["chat_handler"].handle_query_stream(user_input):
                    # 将字典转换为JSON字符串，并添加换行符（SSE格式）
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式生成出错: {str(e)}", exc_info=True)
                error_data = {
                    "type": "error",
                    "message": f"处理出错: {str(e)}"
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        logger.error(f"处理流式请求时出错: {str(e)}")
        return jsonify({"error": f"处理出错: {str(e)}"}), 500


@app.route("/health")
def health():
    """健康检查端点"""
    import psutil
    from datetime import datetime

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    mem_percent = process.memory_percent()
    cpu_percent = psutil.cpu_percent(interval=1)

    # 获取并发状态
    limiter = components["limiter"]
    concurrency_stats = limiter.get_stats()

    status = {
        "status": "healthy",
        "memory_mb": round(mem_mb, 2),
        "memory_percent": round(mem_percent, 2),
        "cpu_percent": round(cpu_percent, 2),
        "concurrency": concurrency_stats,
        "timestamp": datetime.now().isoformat()
    }

    if mem_percent > config.MEMORY_WARNING_THRESHOLD or cpu_percent > 90:
        status["status"] = "warning"
        status["message"] = "资源使用率过高"

    # 如果并发请求过多也发出警告
    if concurrency_stats["active_requests"] >= concurrency_stats["max_concurrent"]:
        status["status"] = "warning"
        if "message" not in status:
            status["message"] = "并发请求已满"
        else:
            status["message"] += "，并发请求已满"

    return jsonify(status)


@app.route("/reload", methods=["POST"])
def reload_documents():
    """手动触发增量加载新文档"""
    if not config.ENABLE_INCREMENTAL_LOAD:
        return jsonify({
            "success": False,
            "message": "增量加载功能未启用"
        }), 400

    try:
        rag_manager = components["rag_manager"]

        # 检查新文档
        new_docs = rag_manager.get_new_documents()
        if not new_docs:
            return jsonify({
                "success": True,
                "message": "没有发现新文档",
                "new_documents": 0
            })

        # 执行增量加载
        new_doc_count = rag_manager.incremental_load()

        logger.info(f"手动增量加载完成: {new_doc_count} 个新文档")

        return jsonify({
            "success": True,
            "message": f"成功加载 {new_doc_count} 个新文档",
            "new_documents": new_doc_count
        })

    except Exception as e:
        logger.error(f"手动增量加载失败: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"加载失败: {str(e)}"
        }), 500


@app.route("/stats/concurrency")
def concurrency_stats():
    """获取并发控制统计信息"""
    limiter = components["limiter"]
    stats = limiter.get_stats()

    # 添加更详细的统计信息
    if stats["total_requests"] > 0:
        stats["reject_rate"] = round(
            stats["rejected_requests"] / stats["total_requests"] * 100, 2
        )
        stats["completion_rate"] = round(
            stats["completed_requests"] / stats["total_requests"] * 100, 2
        )
    else:
        stats["reject_rate"] = 0
        stats["completion_rate"] = 0

    return jsonify(stats)


@app.route("/stats/security")
def security_stats():
    """获取安全过滤器统计信息"""
    try:
        chat_handler = components["chat_handler"]
        if hasattr(chat_handler, 'security_filter') and chat_handler.security_filter:
            stats = chat_handler.security_filter.get_security_stats()
            return jsonify({
                "status": "success",
                "data": stats
            })
        else:
            return jsonify({
                "status": "success",
                "data": {"enabled": False, "message": "安全过滤器未启用"}
            })
    except Exception as e:
        logger.error(f"获取安全统计失败: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"获取安全统计失败: {str(e)}"
        }), 500


@app.route("/pending")
def pending_documents():
    """查看待处理的新文档列表"""
    rag_manager = components["rag_manager"]
    new_docs = rag_manager.get_new_documents()
    doc_list = [os.path.basename(doc) for doc in new_docs]

    return jsonify({
        "count": len(new_docs),
        "documents": doc_list
    })


# 为路由动态应用频率限制
if rate_limiter:
    # 使用 Flask-Limiter 的 exempt_when 或直接重新定义路由
    # 由于装饰器已在路由定义时应用，这里使用 limit_decorator 方法

    # 对话接口 - 应用多个限制
    if isinstance(config.RATE_LIMIT_CHAT, list):
        # 使用 shared_limit 定义命名限制
        chat_limit_key = "chat_endpoint"
        limiter_decorator = rate_limiter.shared_limit(
            ";".join(config.RATE_LIMIT_CHAT),  # 分号分隔多个限制
            scope=chat_limit_key
        )
        # 重新包装路由函数
        chat_stream_original = chat_stream
        chat_stream = limiter_decorator(chat_stream_original)
        # 更新路由
        app.view_functions['chat_stream'] = chat_stream

    # 健康检查接口
    health_original = health
    health = rate_limiter.limit(config.RATE_LIMIT_HEALTH)(health_original)
    app.view_functions['health'] = health

    # 统计接口
    concurrency_stats_original = concurrency_stats
    concurrency_stats = rate_limiter.limit(config.RATE_LIMIT_STATS)(concurrency_stats_original)
    app.view_functions['concurrency_stats'] = concurrency_stats

    security_stats_original = security_stats
    security_stats = rate_limiter.limit(config.RATE_LIMIT_STATS)(security_stats_original)
    app.view_functions['security_stats'] = security_stats

    logger.info("频率限制装饰器已应用到所有路由")


if __name__ == "__main__":
    logger.info(f"启动Flask服务器: {config.HOST}:{config.PORT}")
    app.run(
        debug=config.DEBUG_MODE,
        host=config.HOST,
        port=config.PORT
    )

