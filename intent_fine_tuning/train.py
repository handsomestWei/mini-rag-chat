#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类器训练脚本
"""

import os
import sys
import argparse
from intent_classifier import IntentClassifier
from fine_tuning_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SAVE_DIR,
    DEFAULT_TEST_SIZE
)


def main():
    parser = argparse.ArgumentParser(description='训练意图分类器')
    parser.add_argument('--model_path', type=str, default=DEFAULT_EMBEDDING_MODEL,
                       help='embedding模型路径（默认：m3e-small）')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                       help='模型保存目录')
    parser.add_argument('--test_size', type=float, default=DEFAULT_TEST_SIZE,
                       help='测试集比例')

    args = parser.parse_args()

    print("🚀 开始训练意图分类器...")
    print(f"📁 模型路径: {args.model_path}")
    print(f"💾 保存目录: {args.save_dir}")
    print(f"📊 测试集比例: {args.test_size}")

    # 创建分类器
    classifier = IntentClassifier(model_path=args.model_path)

    # 准备训练数据
    training_data = classifier.prepare_training_data()

    # 训练分类器
    accuracy = classifier.train_classifier(training_data, test_size=args.test_size)

    # 保存模型
    classifier.save_model(args.save_dir)

    print(f"✅ 训练完成！准确率: {accuracy:.3f}")
    print(f"📁 模型已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
