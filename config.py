"""
低配置服务器RAG配置文件
根据你的服务器配置调整以下参数
"""

# ========== 嵌入模型配置 ==========
# 中文优化模型：moka-ai/m3e-small (轻量版)
# - 专为RAG检索优化的中英双语模型
# - 中文语义理解显著优于all-MiniLM-L6-v2
# - 模型更小（200MB vs 400MB），内存占用更低（1GB vs 1.5GB）
# - 适合4GB内存服务器，需要同时运行LLM和其他应用
# - 使用本地已下载的模型（无需联网）
EMBEDDING_MODEL_PATH = "./model/moka-ai/m3e-small"
M3E_MODEL_PATH = EMBEDDING_MODEL_PATH  # 兼容性别名

# CPU配置
EMBEDDING_DEVICE = "cpu"  # cpu 或 cuda:0

# 批处理大小 (影响内存和速度)
# m3e-small推荐: 10-12 (512维向量，比m3e-base的768维更小，可以加大batch)
EMBEDDING_BATCH_SIZE = 10  # m3e-small优化配置

# ========== 向量数据库配置 ==========
VECTOR_STORE_PATH = "./vector_store"

# 检索配置
# k: 返回最相关的文档数量（最终返回给LLM）
# fetch_k: 预筛选的文档数量（从向量库中先取出这么多）
# 平衡优化：适度增加检索范围，提高命中率
RETRIEVER_K = 2  # 返回2个文档，平衡质量和速度
RETRIEVER_FETCH_K = 20  # 预筛选20个，提高准确性

# ========== 文档处理配置 ==========
DATA_PATH = "./data/"
DATA_NEW_PATH = "./data_new/"  # 增量文档目录

# 文档切片大小优化
# 平衡优化：适中的chunk保留完整语义，overlap确保连续性
CHUNK_SIZE = 800  # 800字符，平衡上下文完整性和速度
CHUNK_OVERLAP = 150  # 150字符overlap，确保信息不断裂

# 高级检索配置
# similarity: 纯相似度检索（返回top-k最相似文档，推荐）
# mmr: 最大边际相关性（平衡相似度和多样性）
# similarity_score_threshold: 相似度阈值过滤（⚠️ FAISS不兼容，会导致问题）
SEARCH_TYPE = "similarity"  # 使用相似度检索，返回top-k文档
SCORE_THRESHOLD = 0.5  # 仅在similarity_score_threshold模式下使用（当前未启用）
MMR_DIVERSITY_SCORE = 0.5  # MMR多样性分数（仅在MMR模式下使用）

# 增量加载配置
ENABLE_INCREMENTAL_LOAD = True  # 是否启用增量加载
AUTO_MIGRATE_PROCESSED = True  # 处理成功后自动迁移文档到原目录

# ========== LLM配置 ==========
# Ollama模型选择 - 2核4G服务器推荐配置
#
# 🏆 强烈推荐（中文优先）:
# - "qwen2:1.5b"   : ~1.5GB, 2-4秒响应, 中文能力极强 ⭐⭐⭐⭐⭐
#
# 🥈 推荐（中英平衡）:
# - "phi3:mini"    : ~2.8GB, 3-6秒响应, 质量优秀 ⭐⭐⭐⭐⭐
#
# 其他选项:
# - "gemma:2b"     : ~2.0GB, 2-5秒响应, 英文优秀 ⭐⭐⭐⭐
# - "tinyllama"    : ~1.0GB, 1-3秒响应, 速度优先 ⭐⭐⭐
# - "llama3:8b"    : ~5.5GB, 8-15秒响应, ❌ 对4G内存太大
#
# 详细对比请查看: MODEL_SELECTION_GUIDE.md
LLM_MODEL = "qwen2:1.5b"  # 2核4G最佳选择

# Ollama性能优化参数
OLLAMA_NUM_CTX = 1024  # 上下文窗口（从2048减到1024，提速显著）
OLLAMA_NUM_THREAD = 4  # CPU线程数（2核服务器设为2）
OLLAMA_NUM_PREDICT = 200  # 最大生成token数（150→200，允许更完整回答）
OLLAMA_TEMPERATURE = 0.3  # 温度（0.1→0.3，允许合理总结，避免过于死板）
OLLAMA_TOP_K = 20  # Top-K采样（10→20，增加灵活性）
OLLAMA_TOP_P = 0.8  # Top-P采样（0.7→0.8，允许更自然的表达）
OLLAMA_REPEAT_PENALTY = 1.1  # 重复惩罚（避免重复生成）

# ========== 提示词配置 ==========
# RAG系统提示词模板（可自定义）
SYSTEM_PROMPT = """你是专业的宠物狗知识助手。请根据提供的参考资料回答用户问题。要求：用中文回答，基于参考资料，简洁明了（150字以内），不编造信息。"""

# 用户问题模板
USER_QUESTION_TEMPLATE = """参考资料：
{context}

问题：{question}

请根据以上参考资料回答问题："""

# ========== 查询优化配置 ==========
# 查询扩展：是否启用查询改写
ENABLE_QUERY_EXPANSION = True

# 查询扩展模板：用LLM扩展用户问题为更详细的检索查询
QUERY_EXPANSION_TEMPLATE = """将用户问题改写为3-5个中文关键词，用空格或逗号分隔。

用户问题：{question}

检索关键词（只输出中文关键词，不要序号、不要解释、不要英文）："""

# ========== 文档压缩配置 ==========
# 是否启用文档压缩（压缩检索到的文档，减少LLM上下文）
ENABLE_DOC_COMPRESSION = True

# 每个文档保留的最大句子数
MAX_SENTENCES_PER_DOC = 3  # 推荐3-5句，平衡信息量和压缩率

# 最小句子长度（字符数，过滤掉太短的句子）
MIN_SENTENCE_LENGTH = 5  # 少于5个字的句子会被过滤

# 压缩方法
# - "hybrid" : TextRank初筛 + m3e重排序（推荐，最佳效果）
# - "m3e" : 仅使用m3e重排序（如果没有jieba）
# - "textrank" : 仅使用TextRank（最快，但效果一般）
DOC_COMPRESSION_METHOD = "hybrid"  # 混合方案

# ========== Flask配置 ==========
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000

# ========== 并发控制配置 ==========
# 最大并发对话数（2核4G推荐: 2-3）
MAX_CONCURRENT_REQUESTS = 4  # 同时处理的最大请求数

# 等待队列大小（0 = 禁用队列，直接拒绝）
MAX_QUEUE_SIZE = 4  # 排队等待的最大请求数

# 请求超时时间（秒）
REQUEST_TIMEOUT = 60  # 单个请求最大处理时间

# 是否启用并发限制
ENABLE_CONCURRENCY_LIMIT = True  # 启用并发控制（推荐）

# 拒绝请求时的提示消息
CONCURRENCY_LIMIT_MESSAGE = "当前服务繁忙，请稍后再试"
QUEUE_FULL_MESSAGE = "请求队列已满，请稍后重试"

# ========== 接口频率限制配置 ==========
# 是否启用频率限制
ENABLE_RATE_LIMIT = True  # 启用频率限制（推荐）

# 频率限制规则（Flask-Limiter格式）
# 格式说明：
# - "10 per minute" = 每分钟10次
# - "100 per hour" = 每小时100次
# - "1000 per day" = 每天1000次
# 可以组合多个规则，如 ["10 per minute", "100 per hour"]

# 全局默认限制（应用于所有接口）
RATE_LIMIT_DEFAULT = "60 per minute"  # 每分钟60次请求

# 对话接口限制（更严格，防止脚本攻击）
RATE_LIMIT_CHAT = [
    "1 per minute",   # 每分钟10次
]

# 健康检查接口限制（更宽松）
RATE_LIMIT_HEALTH = "5 per minute"

# 统计接口限制
RATE_LIMIT_STATS = "5 per minute"

# 频率限制存储方式
# - "memory" : 内存存储（重启后重置，适合单机部署，推荐）
# - "redis" : Redis存储（支持分布式，需要Redis服务）
RATE_LIMIT_STORAGE = "memory"  # 使用内存存储（简单轻量）

# Redis配置（仅在 RATE_LIMIT_STORAGE="redis" 时使用）
RATE_LIMIT_REDIS_URL = "redis://localhost:6379/0"

# 频率限制错误消息
RATE_LIMIT_MESSAGE = "请求过于频繁，请稍后再试"

# 是否在响应头中返回限制信息
RATE_LIMIT_HEADERS_ENABLED = True  # 返回 X-RateLimit-* 响应头

# 频率限制策略
# - "fixed-window" : 固定窗口（默认，简单高效）
# - "moving-window" : 滑动窗口（更精确，消耗稍高）
RATE_LIMIT_STRATEGY = "fixed-window"

# 信任的代理服务器（用于获取真实IP）
# 如果部署在 Nginx/CDN 后面，设置为 True
TRUST_PROXY_HEADERS = True  # 信任 X-Forwarded-For 等代理头

# 脚本检测功能
ENABLE_BOT_DETECTION = True  # 启用脚本/机器人检测

# 可疑User-Agent关键词（小写）
SUSPICIOUS_USER_AGENTS = [
    "bot", "crawler", "spider", "scraper", "curl", "wget",
    "python-requests", "python-urllib", "java", "go-http-client",
    "postman", "insomnia", "httpie"
]

# 白名单User-Agent（即使包含上述关键词也放行）
WHITELISTED_USER_AGENTS = [
    "googlebot", "bingbot", "baiduspider"  # 搜索引擎爬虫
]

# 是否阻止空User-Agent
BLOCK_EMPTY_USER_AGENT = True  # 阻止没有User-Agent的请求

# 是否阻止缺少Referer的非首页请求
CHECK_REFERER = False  # 暂时禁用（可能误伤正常用户）

# ========== Web界面配置 ==========
# 页面标题和品牌
WEB_APP_TITLE = "智能问答助手"  # 浏览器标题
WEB_APP_SUBTITLE = "基于 RAG 的知识问答系统"  # 副标题
WEB_HEADER_ICON = "💬"  # 页面头部图标

# 对话图标
WEB_USER_ICON = "👤"  # 用户头像图标
WEB_AI_ICON = "🤖"  # AI头像图标

# 欢迎消息
WEB_WELCOME_TITLE = "👋 欢迎使用智能问答助手"
WEB_WELCOME_MESSAGE = "我可以根据知识库内容回答您的问题"
WEB_WELCOME_HINT = "请在下方输入您的问题开始咨询"

# 主题颜色（CSS颜色值）
WEB_PRIMARY_COLOR = "#667eea"  # 主色调
WEB_SECONDARY_COLOR = "#764ba2"  # 次色调
WEB_ACCENT_COLOR = "#f093fb"  # 强调色
WEB_ACCENT_SECONDARY = "#f5576c"  # 强调色2

# 功能开关
WEB_ENABLE_STREAMING = True  # 是否启用流式输出（推荐开启）

# ========== 性能优化配置 ==========
# 是否在启动时预加载向量库
PRELOAD_VECTOR_STORE = True

# 是否启用进度条
SHOW_PROGRESS_BAR = True

# 内存监控
ENABLE_MEMORY_MONITOR = True
MEMORY_WARNING_THRESHOLD = 85  # 内存使用率警告阈值(%)

# ========== 安全配置 ==========
# 输入长度限制
MAX_INPUT_LENGTH = 100  # 最大输入字符数
ENABLE_INPUT_TRUNCATION = True  # 是否启用输入截断（True=截断，False=拒绝）

# 对话安全防护
ENABLE_SECURITY_FILTER = True  # 是否启用安全过滤器
SECURITY_BLOCKED_MESSAGES = [
    "系统提示词", "system prompt", "system_prompt",
    "模型", "model", "模型信息", "model info", "model_info",
    "知识库", "knowledge base", "knowledge_base",
    "向量库", "vector store", "vector_store",
    "embedding", "嵌入", "embeddings",
    "检索", "retrieval", "retrieve",
    "文档", "documents", "docs",
    "原始数据", "raw data", "source",
    "提示词", "prompt", "template",
    "配置", "config", "configuration",
    "什么模型", "which model", "什么技术", "what technology"
]  # 敏感关键词列表

SECURITY_RESPONSE_TEMPLATE = "抱歉，我无法回答关于系统技术细节的问题。请问您有什么其他需要帮助的吗？"

# 流式输出状态消息配置
STREAM_STATUS_RETRIEVING = "正在检索相关文档..."
STREAM_STATUS_GENERATING = "正在生成回答..."

# 错误消息配置
ERROR_NO_RESPONSE = "服务正忙，请稍后再试"

# ========== 日志配置 ==========
# 日志级别：
# - DEBUG: 详细调试信息（开发/调试时使用）
# - INFO:  一般信息（生产环境推荐）
# - WARNING: 警告信息
# - ERROR: 错误信息
#
# DEBUG 模式会额外打印：
# - 🔍 查询扩展详情（原始查询 -> 扩展查询）
# - 📄 检索到的文档完整内容（包含元数据、字符数）
# - 💭 发送给LLM的完整提示词
# - 💬 LLM生成的完整回答内容
# - 其他详细的性能和内存统计
#
# 使用建议：
# - 开发/调试环境: LOG_LEVEL = "DEBUG"  （可查看所有细节）
# - 生产环境:       LOG_LEVEL = "INFO"   （性能更好，日志更简洁）
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
LOG_PATH = "log"  # 日志输出目录

# ========== 日志管理配置 ==========
# 日志轮转配置
LOG_MAX_SIZE_MB = 10              # 单个日志文件最大大小（MB）
LOG_BACKUP_COUNT = 5              # 保留的备份文件数量
LOG_DAILY_BACKUP_COUNT = 7        # 每日日志保留天数
LOG_MAX_AGE_DAYS = 7             # 日志文件最大保留天数（自动清理）

# ========== 意图识别配置 ==========
# 启用意图识别功能（优化RAG调用）
ENABLE_INTENT_CLASSIFICATION = True

# 意图分类器模型路径
INTENT_MODEL_PATH = "./model/intent-classifier"

# ML模型最低置信度（低于此值时使用默认RAG）
INTENT_ML_MIN_CONFIDENCE = 0.7

# 意图识别规则（正则表达式）
INTENT_RULES = {
    "greeting": [
        r"^(你好|您好|hi|hello|hey|早上好|下午好|晚上好|嗨)[！!。.～~]*$",
        r"^(在吗|在不在|有人吗)[？?]*$",
    ],
    "politeness": [
        r"^(谢谢|感谢|多谢|thanks|thank you)[！!。.～~]*$",
        r"^(再见|拜拜|bye|goodbye|see you)[！!。.～~]*$",
        r"^(不用了|不需要|没事)[。.！!]*$",
    ],
    "system": [
        r".*(系统提示|提示词|prompt|system prompt).*",
        r".*(模型|model|技术栈|技术细节|architecture).*",
        r".*(知识库|向量库|数据库|database).*",
        r".*(你是.*模型|什么模型|which model|哪个模型).*",
    ],
    "unknown": [
        r"^(嗯|啊|哦|呃|额|这个|那个|嘛)[。.！!？?～~]*$",
        r"^[？?]+$",
        r"^[。.！!]+$",
    ]
}

# 意图识别关键词
INTENT_KEYWORDS = {
    "greeting": [
        "你好", "您好", "hi", "hello", "hey", "早上好", "下午好", "晚上好",
        "在吗", "在不在", "有人吗", "嗨"
    ],
    "politeness": [
        "谢谢", "感谢", "多谢", "thanks", "thank you",
        "再见", "拜拜", "bye", "goodbye", "see you",
        "不用了", "不需要", "没事"
    ],
    "system": [
        "系统提示", "提示词", "prompt", "system prompt",
        "模型", "model", "技术栈", "技术细节", "architecture",
        "知识库", "向量库", "数据库", "database",
        "什么模型", "哪个模型", "which model"
    ],
    "unknown": [
        "嗯", "啊", "哦", "呃", "额", "这个", "那个", "嘛"
    ]
}

# 意图响应配置
INTENT_RESPONSES = {
    "greeting": {
        "skip_rag": True,
        "response": "你好！我是智能问答助手，有什么可以帮助您的吗？"
    },
    "politeness": {
        "skip_rag": True,
        "response": "不客气！如有其他问题，随时提问。"
    },
    "system": {
        "skip_rag": True,
        "response": "抱歉，我无法回答关于系统技术细节的问题。请询问知识库相关的内容。"
    },
    "chitchat": {
        "skip_rag": True,
        "response": "抱歉，我是专业的知识问答助手，暂时无法回答时间、天气等闲聊问题。请询问知识库相关的内容。"
    },
    "knowledge": {
        "skip_rag": False,
        "response": None  # 正常RAG流程
    },
    "unknown": {
        "skip_rag": True,
        "response": "抱歉，我还不能理解您的问题，请补充完善您的提问。"
    }
}

