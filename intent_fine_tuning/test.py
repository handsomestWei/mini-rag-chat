#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类器测试脚本
"""

import argparse
from intent_classifier import IntentClassifier, load_test_data


def main():
    parser = argparse.ArgumentParser(description='测试意图分类器')
    parser.add_argument('--model_path', type=str, default='../model/moka-ai/m3e-small',
                       help='m3e-small模型路径')
    parser.add_argument('--model_dir', type=str, default='./models/intent-classifier',
                       help='训练好的模型目录')
    parser.add_argument('--text', type=str, default=None,
                       help='要测试的文本')
    parser.add_argument('--test_file', type=str, default='test_data.json',
                       help='测试数据文件')

    args = parser.parse_args()

    print("🧪 测试意图分类器...")

    # 创建分类器并加载模型
    classifier = IntentClassifier(model_path=args.model_path)
    classifier.load_model(args.model_dir)

    # 加载测试数据
    if args.text:
        # 单文本测试
        test_cases = [{"text": args.text, "expected_intent": "unknown", "description": "用户输入"}]
    else:
        # 从文件加载测试用例
        test_cases = load_test_data(args.test_file)

    print(f"\n📋 加载了 {len(test_cases)} 个测试用例")
    print("="*60)
    print("测试结果:")
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
        print(f"    跳过RAG: {result['skip_rag']}")

        if result['response']:
            print(f"    回复: {result['response']}")

        # 显示概率分布
        print("    📈 概率分布:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for intent, prob in sorted_probs[:3]:  # 只显示前3个
            print(f"       {intent}: {prob:.3f}")

        print("-" * 60)

    if total_count > 1:
        test_accuracy = correct_count / total_count
        print(f"\n📊 总体测试结果:")
        print(f"   总测试用例: {total_count}")
        print(f"   正确预测: {correct_count}")
        print(f"   测试准确率: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()
