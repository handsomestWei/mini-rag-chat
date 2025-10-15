#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别模块 - 混合方案（规则匹配 + 小模型）
"""

import os
import re
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger


class IntentClassifier:
    """
    混合意图识别器

    使用规则匹配 + 小模型的混合方案：
    1. 首先使用规则和关键词匹配（快速、准确）
    2. 如果规则未匹配，使用小模型分类（灵活、智能）
    """

    def __init__(self, config):
        """
        初始化意图识别器

        Args:
            config: 配置对象
        """
        self.config = config
        self.ml_classifier = None
        self.ml_available = False

        # 从配置加载规则和关键词
        self.rules = getattr(config, 'INTENT_RULES', self._get_default_rules())
        self.keywords = getattr(config, 'INTENT_KEYWORDS', self._get_default_keywords())

        # 尝试加载机器学习模型
        self._load_ml_classifier()

        logger.info("意图识别器初始化完成")
        logger.info(f"  - 规则匹配: {len(self.rules)} 个规则")
        logger.info(f"  - 关键词匹配: {sum(len(v) for v in self.keywords.values())} 个关键词")
        logger.info(f"  - 机器学习模型: {'已加载' if self.ml_available else '未加载（使用规则模式）'}")

    def _get_default_rules(self) -> Dict[str, List[str]]:
        """
        获取默认规则（正则表达式）

        Returns:
            Dict: 意图 -> 正则表达式列表
        """
        return {
            "greeting": [
                r"^(你好|您好|hi|hello|hey|早上好|下午好|晚上好|嗨)[！!。.～~]*$",
                r"^(在吗|在不在|有人吗)[？?]*$",
                r"^(你是谁|你是什么|你叫什么|你叫什么名字)[？?]*$",
                r"^(你能做什么|你可以做什么|你有什么功能|你能帮我做什么)[？?]*$",
                r"^(介绍一下你自己|自我介绍|你是干什么的|你是做什么的)[？?]*$",
                r"^(who are you|what are you|what can you do|introduce yourself)[？?]*$",
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
            "chitchat": [
                r"^(现在|几点|什么时间|今天|明天|星期|天气).*[？?]?$",
                r".*(几点了|什么时间|今天星期几|今天几号|现在是几号).*",
                r".*(天气|weather|温度|下雨|晴天).*",
                r".*(讲个笑话|唱首歌|背首诗|猜个谜语|陪我聊天).*",
                r".*(无聊|好无聊|闲着|没事做).*",
                r".*(算.*题|翻译|用.*语怎么说).*",
            ],
            "unknown": [
                r"^(嗯|啊|哦|呃|额|这个|那个|嘛)[。.！!？?～~]*$",
                r"^[？?]+$",
                r"^[。.！!]+$",
            ]
        }

    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """
        获取默认关键词

        Returns:
            Dict: 意图 -> 关键词列表
        """
        return {
            "greeting": [
                "你好", "您好", "hi", "hello", "hey", "早上好", "下午好", "晚上好",
                "在吗", "在不在", "有人吗", "嗨",
                "你是谁", "你是什么", "你叫什么", "你叫什么名字",
                "你能做什么", "你可以做什么", "你有什么功能", "你能帮我做什么",
                "介绍一下你自己", "自我介绍", "你是干什么的", "你是做什么的",
                "who are you", "what are you", "what can you do", "introduce yourself"
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
            "chitchat": [
                "现在几点", "几点了", "什么时间", "现在是几号", "今天星期几", "今天几号",
                "天气怎么样", "今天天气", "明天天气", "weather",
                "讲个笑话", "唱首歌", "背首诗", "猜个谜语", "陪我聊天",
                "无聊", "好无聊", "闲着", "没事做",
                "算一道题", "翻译", "用英语怎么说"
            ],
            "unknown": [
                "嗯", "啊", "哦", "呃", "额", "这个", "那个", "嘛"
            ]
        }

    def _get_default_intent_config(self) -> Dict:
        """
        获取默认意图响应配置

        Returns:
            Dict: 意图 -> 配置字典
        """
        return {
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
                "response": None
            },
            "unknown": {
                "skip_rag": True,
                "response": "抱歉，我还不能理解您的问题，请补充完善您的提问。"
            }
        }

    def _load_ml_classifier(self):
        """加载机器学习分类器"""
        try:
            # 检查模型路径
            model_dir = Path(self.config.INTENT_MODEL_PATH)
            if not model_dir.exists():
                logger.warning(f"意图分类器模型目录不存在: {model_dir}")
                logger.warning("将仅使用规则匹配模式")
                return

            # 动态导入以避免必须依赖
            import sys
            intent_fine_tuning_path = str(Path(__file__).parent.parent / "intent_fine_tuning")
            if intent_fine_tuning_path not in sys.path:
                sys.path.insert(0, intent_fine_tuning_path)

            from intent_classifier import IntentClassifier as MLClassifier

            # 加载模型
            self.ml_classifier = MLClassifier(
                model_path=str(Path(self.config.M3E_MODEL_PATH))
            )
            self.ml_classifier.load_model(str(model_dir))
            self.ml_available = True

            logger.info(f"机器学习分类器加载成功: {model_dir}")

        except ImportError as e:
            logger.warning(f"无法导入意图分类器模块: {e}")
            logger.warning("将仅使用规则匹配模式")
        except Exception as e:
            logger.warning(f"加载机器学习分类器失败: {e}")
            logger.warning("将仅使用规则匹配模式")

    def _match_by_rules(self, text: str) -> Optional[str]:
        """
        使用正则规则匹配

        Args:
            text: 输入文本

        Returns:
            Optional[str]: 匹配到的意图，或None
        """
        text_lower = text.lower().strip()

        for intent, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    logger.debug(f"规则匹配命中: '{text}' -> {intent} (规则: {pattern})")
                    return intent

        return None

    def _match_by_keywords(self, text: str) -> Optional[str]:
        """
        使用关键词匹配

        Args:
            text: 输入文本

        Returns:
            Optional[str]: 匹配到的意图，或None
        """
        text_lower = text.lower().strip()

        # 计算每个意图的匹配分数
        scores = {}
        for intent, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # 完全匹配得分更高
                    if text_lower == keyword.lower():
                        score += 10
                    else:
                        score += 1

            if score > 0:
                scores[intent] = score

        if scores:
            # 返回得分最高的意图
            best_intent = max(scores.items(), key=lambda x: x[1])
            logger.debug(f"关键词匹配命中: '{text}' -> {best_intent[0]} (分数: {best_intent[1]})")
            return best_intent[0]

        return None

    def _classify_by_ml(self, text: str) -> Optional[Dict]:
        """
        使用机器学习模型分类

        Args:
            text: 输入文本

        Returns:
            Optional[Dict]: 分类结果，或None
        """
        if not self.ml_available:
            return None

        try:
            result = self.ml_classifier.classify(text)
            logger.debug(f"ML分类结果: '{text}' -> {result['intent']} (置信度: {result['confidence']:.3f})")
            return result
        except Exception as e:
            logger.error(f"ML分类失败: {e}")
            return None

    def classify(self, text: str) -> Dict:
        """
        对文本进行意图分类（混合方案）

        流程：
        1. 规则匹配（最快、最准确）
        2. 关键词匹配（快速、相对准确）
        3. ML模型分类（灵活、智能）
        4. 默认返回knowledge（需要RAG）

        Args:
            text: 输入文本

        Returns:
            Dict: 分类结果
            {
                "intent": str,           # 意图类别
                "confidence": float,     # 置信度
                "skip_rag": bool,        # 是否跳过RAG
                "response": str,         # 预定义回复（如果有）
                "method": str            # 使用的方法: "rule" | "keyword" | "ml" | "default"
            }
        """
        logger.debug(f"开始意图识别: '{text}'")

        # 1. 规则匹配
        intent = self._match_by_rules(text)
        if intent:
            return self._build_result(intent, 1.0, "rule")

        # 2. 关键词匹配
        intent = self._match_by_keywords(text)
        if intent:
            return self._build_result(intent, 0.9, "keyword")

        # 3. ML模型分类
        ml_result = self._classify_by_ml(text)
        if ml_result:
            # 只有当置信度足够高且不是knowledge时才使用ML结果
            min_confidence = getattr(self.config, 'INTENT_ML_MIN_CONFIDENCE', 0.7)
            if ml_result['confidence'] >= min_confidence and ml_result['intent'] != 'knowledge':
                return self._build_result(
                    ml_result['intent'],
                    ml_result['confidence'],
                    "ml",
                    ml_response=ml_result.get('response')
                )

        # 4. 默认：需要知识问答（使用RAG）
        logger.debug(f"未匹配到特定意图，默认为knowledge: '{text}'")
        return self._build_result("knowledge", 0.5, "default")

    def _build_result(self, intent: str, confidence: float, method: str,
                     ml_response: Optional[str] = None) -> Dict:
        """
        构建分类结果

        Args:
            intent: 意图类别
            confidence: 置信度
            method: 使用的方法
            ml_response: ML模型的预定义回复

        Returns:
            Dict: 分类结果
        """
        # 从配置中获取意图响应配置
        intent_responses = getattr(self.config, 'INTENT_RESPONSES', self._get_default_intent_config())

        config = intent_responses.get(intent, intent_responses.get("knowledge", {
            "skip_rag": False,
            "response": None
        }))

        # 如果ML模型提供了回复，使用ML的回复
        if ml_response:
            config["response"] = ml_response

        return {
            "intent": intent,
            "confidence": confidence,
            "skip_rag": config["skip_rag"],
            "response": config["response"],
            "method": method
        }

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            "ml_available": self.ml_available,
            "rules_count": len(self.rules),
            "keywords_count": sum(len(v) for v in self.keywords.values()),
            "intents": list(set(list(self.rules.keys()) + list(self.keywords.keys())))
        }

