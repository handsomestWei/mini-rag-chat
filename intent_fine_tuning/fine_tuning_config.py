#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类器微调配置文件
集中管理所有路径和参数配置
"""

# ========== 模型路径配置 ==========
# 预训练embedding模型路径
# 可修改为其他sentence-transformers兼容模型：
# - "../model/moka-ai/m3e-small"      (推荐，中文优化，512维)
# - "../model/moka-ai/m3e-base"       (更大更准，768维)
# - "BAAI/bge-small-zh-v1.5"          (轻量快速，512维)
# - "BAAI/bge-base-zh-v1.5"           (高质量，768维)
DEFAULT_EMBEDDING_MODEL = "../model/moka-ai/m3e-small"

# ========== 训练输出配置 ==========
# 分类器保存目录
DEFAULT_SAVE_DIR = "./model/intent-classifier"

# ========== 数据文件配置 ==========
# 通用训练数据文件（问候、礼貌、系统、闲聊等，可跨领域复用）
DEFAULT_GENERAL_DATA = "training_data_general.json"

# 领域训练数据文件（知识问答，根据领域修改）
DEFAULT_DOMAIN_DATA = "training_data_domain.json"

# 测试数据文件
DEFAULT_TEST_DATA = "test_data.json"

# ========== 训练参数配置 ==========
# 测试集比例
DEFAULT_TEST_SIZE = 0.2

# LogisticRegression参数
CLASSIFIER_PARAMS = {
    "random_state": 42,
    "max_iter": 1000,
    "multi_class": "ovr"  # 一对多策略
}

# ========== 意图类别定义 ==========
# 支持的意图类别（顺序很重要，影响标签编码）
INTENT_CLASSES = [
    "greeting",      # 问候/自我介绍
    "knowledge",     # 知识问答
    "politeness",    # 礼貌用语
    "system",        # 系统交互
    "chitchat",      # 闲聊（时间、天气、笑话等）
    "unknown"        # 未知/其他
]

# ========== 意图响应配置 ==========
# 每个意图的默认行为和回复
INTENT_CONFIG = {
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

# ========== 使用说明 ==========
"""
如何修改配置：

1. 更换embedding模型：
   DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
   注意：换模型后需要重新训练分类器

2. 修改意图回复：
   INTENT_CONFIG["greeting"]["response"] = "欢迎使用XXX助手！"

3. 添加新的意图类别：
   a. 在 INTENT_CLASSES 中添加新类别
   b. 在 INTENT_CONFIG 中添加配置
   c. 在训练数据文件中添加样本
   d. 重新训练模型

4. 调整训练参数：
   CLASSIFIER_PARAMS["max_iter"] = 2000
   DEFAULT_TEST_SIZE = 0.3
"""

