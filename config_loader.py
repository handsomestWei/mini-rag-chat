"""
配置加载器 - 从 YAML 文件加载配置并转换为类似 config.py 的对象
保持与现有代码的向后兼容性
"""

import os
import yaml
from types import SimpleNamespace
from loguru import logger


class ConfigLoader:
    """配置加载器，将 YAML 配置转换为对象"""
    
    def __init__(self, config_path=None):
        """
        初始化配置加载器
        
        Args:
            config_path: YAML 配置文件路径，默认从环境变量或默认路径读取
        """
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config.yaml")
        
        self.config_path = config_path
        self._config = None
        self.load()
    
    def load(self):
        """加载 YAML 配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            self._config = self._convert_to_object(yaml_config)
            self._apply_env_overrides()
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def _convert_to_object(self, yaml_dict):
        if isinstance(yaml_dict, dict):
            obj = SimpleNamespace()
            for key, value in yaml_dict.items():
                converted_value = self._convert_to_object(value)
                setattr(obj, key, converted_value)
            return obj
        elif isinstance(yaml_dict, list):
            return [self._convert_to_object(item) for item in yaml_dict]
        else:
            return yaml_dict
    
    def _object_to_dict(self, obj):
        """将 SimpleNamespace 对象转换为字典"""
        if isinstance(obj, SimpleNamespace):
            result = {}
            for key in dir(obj):
                if not key.startswith('_'):
                    value = getattr(obj, key)
                    result[key] = self._object_to_dict(value)
            return result
        elif isinstance(obj, list):
            return [self._object_to_dict(item) for item in obj]
        else:
            return obj
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        # Ollama URL 可以从环境变量覆盖
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url and hasattr(self._config, 'ollama'):
            self._config.ollama.base_url = ollama_url
    
    def _flatten_config(self, prefix=''):
        """将嵌套配置扁平化为点号分隔的属性"""
        config_dict = {}
        
        # 扁平化嵌套配置
        # EMBEDDING_MODEL_PATH = embedding.model_path
        if hasattr(self._config, 'embedding'):
            config_dict['EMBEDDING_MODEL_PATH'] = self._config.embedding.model_path
            config_dict['M3E_MODEL_PATH'] = self._config.embedding.model_path
            config_dict['EMBEDDING_DEVICE'] = self._config.embedding.device
            config_dict['EMBEDDING_BATCH_SIZE'] = self._config.embedding.batch_size
        
        # 向量数据库配置
        if hasattr(self._config, 'vector_store'):
            config_dict['VECTOR_STORE_PATH'] = self._config.vector_store.path
            config_dict['RETRIEVER_K'] = self._config.vector_store.retriever_k
            config_dict['RETRIEVER_FETCH_K'] = self._config.vector_store.retriever_fetch_k
            config_dict['SEARCH_TYPE'] = self._config.vector_store.search_type
            config_dict['SCORE_THRESHOLD'] = self._config.vector_store.score_threshold
            config_dict['MMR_DIVERSITY_SCORE'] = self._config.vector_store.mmr_diversity_score
        
        # 文档配置
        if hasattr(self._config, 'document'):
            config_dict['DATA_PATH'] = self._config.document.data_path
            config_dict['DATA_NEW_PATH'] = self._config.document.data_new_path
            config_dict['CHUNK_SIZE'] = self._config.document.chunk_size
            config_dict['CHUNK_OVERLAP'] = self._config.document.chunk_overlap
            config_dict['ENABLE_INCREMENTAL_LOAD'] = self._config.document.enable_incremental_load
            config_dict['AUTO_MIGRATE_PROCESSED'] = self._config.document.auto_migrate_processed
        
        # LLM 提供商配置
        config_dict['LLM_PROVIDER'] = getattr(self._config, 'provider', 'ollama')
        
        # Ollama 配置
        if hasattr(self._config, 'ollama'):
            config_dict['LLM_MODEL'] = self._config.ollama.model
            config_dict['OLLAMA_BASE_URL'] = self._config.ollama.base_url
            config_dict['OLLAMA_NUM_CTX'] = self._config.ollama.num_ctx
            config_dict['OLLAMA_NUM_THREAD'] = self._config.ollama.num_thread
            config_dict['OLLAMA_NUM_PREDICT'] = self._config.ollama.num_predict
            config_dict['OLLAMA_TEMPERATURE'] = self._config.ollama.temperature
            config_dict['OLLAMA_TOP_K'] = self._config.ollama.top_k
            config_dict['OLLAMA_TOP_P'] = self._config.ollama.top_p
            config_dict['OLLAMA_REPEAT_PENALTY'] = self._config.ollama.repeat_penalty
        
        # 线上API服务配置
        if hasattr(self._config, 'online'):
            config_dict['ONLINE_SERVICE'] = self._config.online.service
            config_dict['ONLINE_BASE_URL'] = self._config.online.base_url
            # API Key 优先从环境变量读取，其次从配置文件读取
            config_dict['ONLINE_API_KEY'] = os.getenv('ONLINE_API_KEY') or os.getenv('OLLAMA_API_KEY') or getattr(self._config.online, 'api_key', '') or ''
            config_dict['ONLINE_MODEL'] = self._config.online.model
            config_dict['ONLINE_TEMPERATURE'] = self._config.online.temperature
            config_dict['ONLINE_MAX_TOKENS'] = self._config.online.max_tokens
            config_dict['ONLINE_TOP_P'] = self._config.online.top_p
        
        # 超时配置（适用于所有服务商）
        if hasattr(self._config, 'timeout'):
            config_dict['OLLAMA_REQUEST_TIMEOUT'] = self._config.timeout.request_timeout
            config_dict['OLLAMA_RESPONSE_TIMEOUT'] = self._config.timeout.response_timeout
        else:
            # 向后兼容：如果没有TIMEOUT配置，使用默认值
            config_dict['OLLAMA_REQUEST_TIMEOUT'] = 120
            config_dict['OLLAMA_RESPONSE_TIMEOUT'] = 300
        
        # 提示词配置
        if hasattr(self._config, 'prompt'):
            config_dict['SYSTEM_PROMPT'] = self._config.prompt.system_prompt
            config_dict['USER_QUESTION_TEMPLATE'] = self._config.prompt.user_question_template
        
        # 查询配置
        if hasattr(self._config, 'query'):
            config_dict['ENABLE_QUERY_EXPANSION'] = self._config.query.enable_expansion
            config_dict['QUERY_EXPANSION_TEMPLATE'] = self._config.query.expansion_template
        
        # 文档压缩配置
        if hasattr(self._config, 'compression'):
            config_dict['ENABLE_DOC_COMPRESSION'] = self._config.compression.enable
            config_dict['MAX_SENTENCES_PER_DOC'] = self._config.compression.max_sentences_per_doc
            config_dict['MIN_SENTENCE_LENGTH'] = self._config.compression.min_sentence_length
            config_dict['DOC_COMPRESSION_METHOD'] = self._config.compression.method
        
        # Flask 配置
        if hasattr(self._config, 'flask'):
            config_dict['DEBUG_MODE'] = self._config.flask.debug_mode
            config_dict['HOST'] = self._config.flask.host
            config_dict['PORT'] = self._config.flask.port
        
        # 并发控制配置
        if hasattr(self._config, 'concurrency'):
            config_dict['MAX_CONCURRENT_REQUESTS'] = self._config.concurrency.max_concurrent_requests
            config_dict['MAX_QUEUE_SIZE'] = self._config.concurrency.max_queue_size
            config_dict['REQUEST_TIMEOUT'] = self._config.concurrency.request_timeout
            config_dict['ENABLE_CONCURRENCY_LIMIT'] = self._config.concurrency.enable_limit
            config_dict['CONCURRENCY_LIMIT_MESSAGE'] = self._config.concurrency.limit_message
            config_dict['QUEUE_FULL_MESSAGE'] = self._config.concurrency.queue_full_message
        
        # 频率限制配置
        if hasattr(self._config, 'rate_limit'):
            config_dict['ENABLE_RATE_LIMIT'] = self._config.rate_limit.enable
            config_dict['RATE_LIMIT_DEFAULT'] = self._config.rate_limit.default
            config_dict['RATE_LIMIT_CHAT'] = self._config.rate_limit.chat
            config_dict['RATE_LIMIT_HEALTH'] = self._config.rate_limit.health
            config_dict['RATE_LIMIT_STATS'] = self._config.rate_limit.stats
            config_dict['RATE_LIMIT_STORAGE'] = self._config.rate_limit.storage
            config_dict['RATE_LIMIT_REDIS_URL'] = self._config.rate_limit.redis_url
            config_dict['RATE_LIMIT_MESSAGE'] = self._config.rate_limit.message
            config_dict['RATE_LIMIT_HEADERS_ENABLED'] = self._config.rate_limit.headers_enabled
            config_dict['RATE_LIMIT_STRATEGY'] = self._config.rate_limit.strategy
        
        # 安全配置
        if hasattr(self._config, 'security'):
            config_dict['TRUST_PROXY_HEADERS'] = self._config.security.trust_proxy_headers
            config_dict['ENABLE_BOT_DETECTION'] = self._config.security.enable_bot_detection
            config_dict['SUSPICIOUS_USER_AGENTS'] = self._config.security.suspicious_user_agents
            config_dict['WHITELISTED_USER_AGENTS'] = self._config.security.whitelisted_user_agents
            config_dict['BLOCK_EMPTY_USER_AGENT'] = self._config.security.block_empty_user_agent
            config_dict['CHECK_REFERER'] = self._config.security.check_referer
            config_dict['MAX_INPUT_LENGTH'] = self._config.security.max_input_length
            config_dict['ENABLE_INPUT_TRUNCATION'] = self._config.security.enable_input_truncation
            config_dict['ENABLE_SECURITY_FILTER'] = self._config.security.enable_security_filter
            config_dict['SECURITY_BLOCKED_MESSAGES'] = self._config.security.blocked_messages
            config_dict['SECURITY_RESPONSE_TEMPLATE'] = self._config.security.security_response_template
        
        # Web 配置
        if hasattr(self._config, 'web'):
            config_dict['WEB_APP_TITLE'] = self._config.web.app_title
            config_dict['WEB_APP_SUBTITLE'] = self._config.web.app_subtitle
            config_dict['WEB_HEADER_ICON'] = self._config.web.header_icon
            config_dict['WEB_USER_ICON'] = self._config.web.user_icon
            config_dict['WEB_AI_ICON'] = self._config.web.ai_icon
            config_dict['WEB_WELCOME_TITLE'] = self._config.web.welcome_title
            config_dict['WEB_WELCOME_MESSAGE'] = self._config.web.welcome_message
            config_dict['WEB_WELCOME_HINT'] = self._config.web.welcome_hint
            config_dict['WEB_PRIMARY_COLOR'] = self._config.web.primary_color
            config_dict['WEB_SECONDARY_COLOR'] = self._config.web.secondary_color
            config_dict['WEB_ACCENT_COLOR'] = self._config.web.accent_color
            config_dict['WEB_ACCENT_SECONDARY'] = self._config.web.accent_secondary
            config_dict['WEB_ENABLE_STREAMING'] = self._config.web.enable_streaming
            config_dict['WEB_FOOTER_ENABLE'] = getattr(self._config.web, 'footer_enable', True)
            config_dict['WEB_FOOTER_TEXT'] = getattr(self._config.web, 'footer_text', '由')
            config_dict['WEB_FOOTER_TECH_PROVIDER'] = getattr(self._config.web, 'footer_tech_provider', '百度')
            config_dict['WEB_FOOTER_TECH_URL'] = getattr(self._config.web, 'footer_tech_url', 'https://www.baidu.com')
            config_dict['WEB_FOOTER_SUFFIX'] = getattr(self._config.web, 'footer_suffix', '提供技术支持')
        
        # 性能配置
        if hasattr(self._config, 'performance'):
            config_dict['PRELOAD_VECTOR_STORE'] = self._config.performance.preload_vector_store
            config_dict['SHOW_PROGRESS_BAR'] = self._config.performance.show_progress_bar
            config_dict['ENABLE_MEMORY_MONITOR'] = self._config.performance.enable_memory_monitor
            config_dict['MEMORY_WARNING_THRESHOLD'] = self._config.performance.memory_warning_threshold
        
        # 流式输出配置
        if hasattr(self._config, 'stream'):
            config_dict['STREAM_STATUS_RETRIEVING'] = self._config.stream.status_retrieving
            config_dict['STREAM_STATUS_GENERATING'] = self._config.stream.status_generating
        
        # 错误配置
        if hasattr(self._config, 'error'):
            config_dict['ERROR_NO_RESPONSE'] = self._config.error.no_response
        
        # 日志配置
        if hasattr(self._config, 'logging'):
            config_dict['LOG_LEVEL'] = self._config.logging.level
            config_dict['LOG_PATH'] = self._config.logging.path
            config_dict['LOG_MAX_SIZE_MB'] = self._config.logging.max_size_mb
            config_dict['LOG_BACKUP_COUNT'] = self._config.logging.backup_count
            config_dict['LOG_DAILY_BACKUP_COUNT'] = self._config.logging.daily_backup_count
            config_dict['LOG_MAX_AGE_DAYS'] = self._config.logging.max_age_days
        
        # 意图识别配置
        if hasattr(self._config, 'intent'):
            config_dict['ENABLE_INTENT_CLASSIFICATION'] = self._config.intent.enable_classification
            config_dict['INTENT_MODEL_PATH'] = self._config.intent.model_path
            config_dict['INTENT_ML_MIN_CONFIDENCE'] = self._config.intent.ml_min_confidence
            if hasattr(self._config.intent, 'rules'):
                config_dict['INTENT_RULES'] = self._object_to_dict(self._config.intent.rules)
            if hasattr(self._config.intent, 'keywords'):
                config_dict['INTENT_KEYWORDS'] = self._object_to_dict(self._config.intent.keywords)
            if hasattr(self._config.intent, 'responses'):
                responses_dict = self._object_to_dict(self._config.intent.responses)
                config_dict['INTENT_RESPONSES'] = responses_dict
        
        return config_dict
    
    def get_config(self):
        """获取配置对象（扁平化，与 config.py 兼容）"""
        config_dict = self._flatten_config()
        config_obj = SimpleNamespace(**config_dict)
        return config_obj


# 全局配置实例
_config_loader = None

def load_config(config_path=None):
    """加载配置文件（单例模式）"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader.get_config()


# 为了保持向后兼容，创建一个 config 对象
# 这样 import config 仍然可以工作
import sys

# 加载配置
config = load_config()

# 将 config 模块注入到 sys.modules，使得 import config 可以工作
sys.modules['config'] = config

