#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类器 - 基于m3e-small的轻量级意图识别
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntentClassifier:
    """基于m3e-small的意图分类器"""

    def __init__(self, model_path: str = "../model/moka-ai/m3e-small"):
        """
        初始化意图分类器

        Args:
            model_path: m3e-small模型路径
        """
        self.model_path = model_path
        self.embedding_model = None
        self.classifier = None
        self.label_encoder = None

        # 意图类别定义
        self.intent_classes = [
            "greeting",      # 问候
            "knowledge",     # 知识问答
            "politeness",    # 礼貌用语
            "system",        # 系统交互
            "chitchat",      # 闲聊（时间、天气、笑话等）
            "unknown"        # 未知/其他
        ]

        # 意图配置
        self.intent_config = {
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

        self._load_embedding_model()

    def _load_embedding_model(self):
        """加载embedding模型"""
        try:
            logger.info(f"正在加载embedding模型: {self.model_path}")
            self.embedding_model = SentenceTransformer(self.model_path)
            logger.info("embedding模型加载完成")
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            raise

    def load_training_data(self,
                          general_file: str = "training_data_general.json",
                          domain_file: str = "training_data_domain.json") -> List[Tuple[str, str]]:
        """
        从外部文件加载训练数据（支持通用数据和领域数据分离）

        Args:
            general_file: 通用训练数据文件路径（问候、礼貌、系统、闲聊等）
            domain_file: 领域训练数据文件路径（知识问答）

        Returns:
            List[Tuple[str, str]]: (文本, 意图标签) 的列表
        """
        training_data = []

        # 加载通用数据
        try:
            with open(general_file, 'r', encoding='utf-8') as f:
                general_data = json.load(f)

            for intent, texts in general_data.items():
                if intent in self.intent_classes:
                    for text in texts:
                        training_data.append((text, intent))

            logger.info(f"从 {general_file} 加载了 {len(training_data)} 条通用数据")
        except FileNotFoundError:
            logger.warning(f"通用数据文件 {general_file} 不存在，跳过")
        except Exception as e:
            logger.error(f"加载通用数据失败: {e}")
            raise

        # 加载领域数据
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)

            domain_count = 0
            for intent, texts in domain_data.items():
                if intent in self.intent_classes:
                    for text in texts:
                        training_data.append((text, intent))
                        domain_count += 1

            logger.info(f"从 {domain_file} 加载了 {domain_count} 条领域数据")
        except FileNotFoundError:
            logger.warning(f"领域数据文件 {domain_file} 不存在，跳过")
        except Exception as e:
            logger.error(f"加载领域数据失败: {e}")
            raise

        if len(training_data) == 0:
            logger.error("未加载到任何训练数据")
            raise ValueError("未加载到任何训练数据，请检查数据文件")

        logger.info(f"总共加载了 {len(training_data)} 条训练数据")
        return training_data


    def prepare_training_data(self,
                             general_file: str = "training_data_general.json",
                             domain_file: str = "training_data_domain.json") -> List[Tuple[str, str]]:
        """
        准备训练数据（兼容性方法）

        Args:
            general_file: 通用训练数据文件路径
            domain_file: 领域训练数据文件路径

        Returns:
            List[Tuple[str, str]]: (文本, 意图标签) 的列表
        """
        return self.load_training_data(general_file, domain_file)

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        提取文本特征（embedding）

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: 特征矩阵
        """
        logger.info(f"正在提取 {len(texts)} 个文本的特征...")

        # 使用tqdm显示进度条
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"特征提取完成，维度: {embeddings.shape}")
        return embeddings

    def train_classifier(self, training_data: List[Tuple[str, str]], test_size: float = 0.2):
        """
        训练分类器

        Args:
            training_data: 训练数据
            test_size: 测试集比例
        """
        logger.info("开始训练意图分类器...")

        # 分离文本和标签
        texts, labels = zip(*training_data)
        texts = list(texts)
        labels = list(labels)

        # 创建标签编码器
        self.label_encoder = {label: i for i, label in enumerate(self.intent_classes)}
        self.id_to_label = {i: label for label, i in self.label_encoder.items()}

        # 转换标签为数字
        y = [self.label_encoder[label] for label in labels]

        # 提取特征
        X = self.extract_features(texts)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 训练分类器
        logger.info("训练LogisticRegression分类器...")
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'  # 一对多策略
        )

        # 使用tqdm显示训练进度（LogisticRegression本身不直接支持进度条，但我们可以模拟）
        logger.info("开始训练...")
        self.classifier.fit(X_train, y_train)
        logger.info("训练完成！")

        # 评估模型
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"训练完成！准确率: {accuracy:.3f}")
        logger.info("分类报告:")
        logger.info(classification_report(y_test, y_pred, target_names=self.intent_classes))

        return accuracy

    def classify(self, text: str) -> Dict[str, any]:
        """
        对文本进行意图分类

        Args:
            text: 输入文本

        Returns:
            Dict: 分类结果
        """
        if not self.classifier:
            raise ValueError("分类器尚未训练，请先调用 train_classifier()")

        # 提取特征
        embedding = self.extract_features([text])

        # 预测
        intent_id = int(self.classifier.predict(embedding)[0])
        intent = self.id_to_label[intent_id]

        # 获取预测概率
        probabilities = self.classifier.predict_proba(embedding)[0]
        confidence = float(np.max(probabilities))

        # 获取配置
        config = self.intent_config[intent]

        result = {
            "intent": intent,
            "confidence": confidence,
            "skip_rag": config["skip_rag"],
            "response": config["response"],
            "probabilities": {
                self.id_to_label[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

        return result

    def save_model(self, model_dir: str = "./models/intent-classifier"):
        """
        保存模型

        Args:
            model_dir: 模型保存目录
        """
        os.makedirs(model_dir, exist_ok=True)

        # 保存分类器
        classifier_path = os.path.join(model_dir, "intent_classifier.pkl")
        joblib.dump(self.classifier, classifier_path)

        # 保存标签编码器（将id_to_label的整数键转换为字符串以便JSON序列化）
        encoder_path = os.path.join(model_dir, "label_encoder.json")
        with open(encoder_path, 'w', encoding='utf-8') as f:
            json.dump({
                "label_to_id": self.label_encoder,
                "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
                "intent_classes": self.intent_classes,
                "intent_config": self.intent_config
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"模型已保存到: {model_dir}")

    def load_model(self, model_dir: str = "./models/intent-classifier"):
        """
        加载模型

        Args:
            model_dir: 模型目录
        """
        # 加载分类器
        classifier_path = os.path.join(model_dir, "intent_classifier.pkl")
        self.classifier = joblib.load(classifier_path)

        # 加载标签编码器（将JSON中的字符串键转换回整数）
        encoder_path = os.path.join(model_dir, "label_encoder.json")
        with open(encoder_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.label_encoder = data["label_to_id"]
            self.id_to_label = {int(k): v for k, v in data["id_to_label"].items()}
            self.intent_classes = data["intent_classes"]
            self.intent_config = data["intent_config"]

        logger.info(f"模型已从 {model_dir} 加载完成")


def load_test_data(test_file: str = "test_data.json") -> List[Dict]:
    """
    从外部文件加载测试数据

    Args:
        test_file: 测试数据文件路径

    Returns:
        List[Dict]: 测试用例列表
    """
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = []
        # 合并所有测试用例
        if 'test_cases' in data:
            test_cases.extend(data['test_cases'])
        if 'edge_cases' in data:
            test_cases.extend(data['edge_cases'])
        if 'mixed_language' in data:
            test_cases.extend(data['mixed_language'])

        logger.info(f"从 {test_file} 加载了 {len(test_cases)} 个测试用例")
        return test_cases

    except FileNotFoundError:
        logger.error(f"测试数据文件 {test_file} 不存在")
        raise
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        raise


def main():
    """主函数 - 训练和测试意图分类器"""
    # 创建分类器
    classifier = IntentClassifier()

    # 准备训练数据
    training_data = classifier.prepare_training_data()

    # 训练分类器
    accuracy = classifier.train_classifier(training_data)

    # 保存模型
    classifier.save_model()

    # 加载测试数据
    test_cases = load_test_data()

    print("\n" + "="*60)
    print("测试分类器:")
    print("="*60)

    correct_count = 0
    total_count = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        expected = test_case.get('expected_intent', 'unknown')
        description = test_case.get('description', '')

        result = classifier.classify(text)
        predicted = result['intent']
        is_correct = predicted == expected

        if is_correct:
            correct_count += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} [{i:2d}/{total_count}] 输入: '{text}'")
        print(f"    预期: {expected:12} | 预测: {predicted:12} | 置信度: {result['confidence']:.3f}")
        print(f"    说明: {description}")
        if result['response']:
            print(f"    回复: {result['response']}")
        print("-" * 60)

    test_accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n📊 测试结果:")
    print(f"   总测试用例: {total_count}")
    print(f"   正确预测: {correct_count}")
    print(f"   测试准确率: {test_accuracy:.3f}")
    print(f"   训练准确率: {accuracy:.3f}")


if __name__ == "__main__":
    main()
