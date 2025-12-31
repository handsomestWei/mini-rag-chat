"""
向量库数据质量评估工具
用于评估向量化质量和为相似度阈值设定提供参考

使用方法：
    python tool/evaluate_quality.py [选项]

选项：
    --top-k N          返回前N个结果（默认：10）
    --output FILE      保存评估报告到文件
    --verbose          显示详细信息
    --help             显示帮助信息

示例：
    # 使用默认测试用例评估
    python tool/evaluate_quality.py

    # 返回前20个结果
    python tool/evaluate_quality.py --top-k 20

    # 保存报告
    python tool/evaluate_quality.py --output evaluation_report.txt

    # 详细模式
    python tool/evaluate_quality.py --verbose

自定义测试用例：
    直接编辑本文件顶部的 TEST_CASES 列表即可
"""

import sys
import io
import os
import argparse
from datetime import datetime

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
config = load_config()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings('ignore')


# ========== 测试用例配置 ==========
# 可以直接在这里修改测试问题和期望关键词

TEST_CASES = [
    # 疾病相关
    {
        "question": "狗狗烧烫伤怎么办",
        "keywords": ["烧烫伤", "护理", "点滴", "抗生素"]
    },

    # 日常护理
    {
        "question": "养狗前的须知",
        "keywords": ["养狗", "准备", "注意事项"]
    }
]

# 如果你想添加新的测试用例，按以下格式添加：
# {
#     "question": "你的问题",
#     "keywords": ["关键词1", "关键词2", "关键词3"]
# },


class VectorStoreEvaluator:
    """向量库质量评估器"""

    def __init__(self, config_obj, top_k=10):
        """
        初始化评估器

        Args:
            config_obj: 配置对象
            top_k: 返回前N个结果
        """
        self.config = config_obj
        self.top_k = top_k
        self.embeddings = None
        self.vector_store = None

        # 加载模型和向量库
        self._load_models()

    def _load_models(self):
        """加载嵌入模型和向量库"""
        print("📦 加载嵌入模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": self.config.EMBEDDING_BATCH_SIZE,
                "normalize_embeddings": True
            }
        )
        print(f"✅ 模型加载完成: {self.config.EMBEDDING_MODEL_PATH}")

        print("\n📚 加载向量库...")
        if not os.path.exists(self.config.VECTOR_STORE_PATH):
            print(f"❌ 向量库不存在: {self.config.VECTOR_STORE_PATH}")
            print("💡 请先运行应用创建向量库: python app_new.py")
            sys.exit(1)

        self.vector_store = FAISS.load_local(
            self.config.VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 获取向量库统计
        total_docs = len(self.vector_store.docstore._dict)
        print(f"✅ 向量库加载完成: {total_docs} 个文档块")

    def evaluate_query(self, question, expected_keywords=None, verbose=False):
        """
        评估单个查询的质量

        Args:
            question: 测试问题
            expected_keywords: 期望的关键词列表（用于验证）
            verbose: 是否显示详细信息

        Returns:
            evaluation: 评估结果字典
        """
        print("\n" + "=" * 80)
        print(f"🔍 问题: {question}")
        if expected_keywords:
            print(f"📌 期望关键词: {', '.join(expected_keywords)}")
        print("=" * 80)

        # 执行相似度搜索
        docs_with_scores = self.vector_store.similarity_search_with_score(
            question,
            k=self.top_k
        )

        # 分析结果
        results = []
        keyword_hits = {kw: [] for kw in (expected_keywords or [])}

        for i, (doc, distance) in enumerate(docs_with_scores, 1):
            # FAISS L2距离转相似度: similarity = 1 / (1 + distance)
            similarity = 1 / (1 + distance)

            # 检查是否包含期望关键词
            hits = []
            if expected_keywords:
                for keyword in expected_keywords:
                    if keyword in doc.page_content:
                        hits.append(keyword)
                        keyword_hits[keyword].append(i)

            result = {
                'rank': i,
                'distance': distance,
                'similarity': similarity,
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'keyword_hits': hits
            }
            results.append(result)

            # 显示结果
            content_preview = doc.page_content[:100].replace('\n', ' ')

            print(f"\n【第 {i} 名】")
            print(f"  🎯 L2距离: {distance:.4f}")
            print(f"  📈 相似度: {similarity:.4f}")

            if hits:
                print(f"  ✅ 命中关键词: {', '.join(hits)}")
            elif expected_keywords:
                print(f"  ❌ 未命中关键词")

            print(f"  📄 内容: {content_preview}...")

            if verbose:
                print(f"  📋 来源: {result['source']}")
                if len(doc.page_content) > 100:
                    print(f"  📝 完整内容:\n{doc.page_content[:300]}...")

        # 分析和建议
        evaluation = self._analyze_results(
            question,
            results,
            keyword_hits,
            expected_keywords
        )

        return evaluation

    def _analyze_results(self, question, results, keyword_hits, expected_keywords):
        """分析评估结果并提供建议"""
        print("\n" + "=" * 80)
        print("📊 评估分析")
        print("=" * 80)

        # 相似度统计
        similarities = [r['similarity'] for r in results]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0

        print(f"\n📈 相似度统计:")
        print(f"  - 最高相似度: {max_similarity:.4f}")
        print(f"  - 平均相似度: {avg_similarity:.4f}")
        print(f"  - Top-3平均: {sum(similarities[:3])/3:.4f}")
        print(f"  - Top-5平均: {sum(similarities[:5])/5:.4f}")

        # 关键词命中分析
        if expected_keywords:
            print(f"\n🎯 关键词命中分析:")
            for keyword, ranks in keyword_hits.items():
                if ranks:
                    best_rank = min(ranks)
                    print(f"  ✅ '{keyword}': 最佳排名第 {best_rank} 位 "
                          f"(共 {len(ranks)} 个块包含)")
                else:
                    print(f"  ❌ '{keyword}': 未在前{self.top_k}名中找到")

        # 质量评级
        print(f"\n⭐ 质量评级:")
        quality_score, quality_grade = self._calculate_quality_score(
            max_similarity,
            avg_similarity,
            keyword_hits,
            expected_keywords
        )
        print(f"  - 综合得分: {quality_score:.2f}/100")
        print(f"  - 质量等级: {quality_grade}")

        # 阈值建议
        print(f"\n💡 阈值建议:")
        threshold_suggestions = self._suggest_thresholds(similarities)
        for mode, threshold in threshold_suggestions.items():
            print(f"  - {mode}: {threshold:.2f}")

        # 改进建议
        print(f"\n🔧 改进建议:")
        suggestions = self._generate_suggestions(
            max_similarity,
            avg_similarity,
            keyword_hits,
            expected_keywords
        )
        for suggestion in suggestions:
            print(f"  • {suggestion}")

        print("=" * 80)

        return {
            'question': question,
            'results': results,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'threshold_suggestions': threshold_suggestions,
            'keyword_hits': keyword_hits
        }

    def _calculate_quality_score(self, max_sim, avg_sim, keyword_hits, expected_keywords):
        """计算质量得分（0-100）"""
        score = 0

        # 最高相似度（40分）
        if max_sim >= 0.7:
            score += 40
        elif max_sim >= 0.6:
            score += 30
        elif max_sim >= 0.5:
            score += 20
        elif max_sim >= 0.4:
            score += 10

        # 平均相似度（30分）
        if avg_sim >= 0.6:
            score += 30
        elif avg_sim >= 0.5:
            score += 20
        elif avg_sim >= 0.4:
            score += 10

        # 关键词命中（30分）
        if expected_keywords:
            hit_count = sum(1 for ranks in keyword_hits.values() if ranks)
            hit_rate = hit_count / len(expected_keywords)

            # 命中率得分（15分）
            score += hit_rate * 15

            # 排名得分（15分）
            if hit_count > 0:
                avg_rank = sum(min(ranks) for ranks in keyword_hits.values() if ranks) / hit_count
                if avg_rank <= 3:
                    score += 15
                elif avg_rank <= 5:
                    score += 10
                elif avg_rank <= 10:
                    score += 5

        # 质量等级
        if score >= 80:
            grade = "优秀 ⭐⭐⭐⭐⭐"
        elif score >= 60:
            grade = "良好 ⭐⭐⭐⭐"
        elif score >= 40:
            grade = "一般 ⭐⭐⭐"
        elif score >= 20:
            grade = "较差 ⭐⭐"
        else:
            grade = "很差 ⭐"

        return score, grade

    def _suggest_thresholds(self, similarities):
        """根据相似度分布建议阈值"""
        if not similarities:
            return {}

        top3_avg = sum(similarities[:3]) / 3
        top5_avg = sum(similarities[:5]) / 5
        max_sim = max(similarities)

        return {
            "严格模式（高精准）": max(0.3, top3_avg - 0.05),
            "平衡模式（推荐）": max(0.25, top5_avg - 0.08),
            "宽松模式（高召回）": max(0.2, top5_avg - 0.15)
        }

    def _generate_suggestions(self, max_sim, avg_sim, keyword_hits, expected_keywords):
        """生成改进建议"""
        suggestions = []

        # 相似度分析
        if max_sim < 0.5:
            suggestions.append("⚠️ 最高相似度过低 - 建议重建向量库或优化文档分块")

        if avg_sim < 0.4:
            suggestions.append("⚠️ 平均相似度过低 - 检索质量差，建议优化数据质量")

        # 关键词分析
        if expected_keywords:
            not_found = [kw for kw, ranks in keyword_hits.items() if not ranks]
            if not_found:
                suggestions.append(f"❌ 关键词未命中: {', '.join(not_found)} - 检查文档是否包含相关内容")

            low_rank = [kw for kw, ranks in keyword_hits.items() if ranks and min(ranks) > 10]
            if low_rank:
                suggestions.append(f"⚠️ 关键词排名靠后: {', '.join(low_rank)} - 考虑减小CHUNK_SIZE")

        # 通用建议
        if max_sim >= 0.6 and avg_sim >= 0.5:
            suggestions.append("✅ 向量库质量良好，可以使用当前配置")

        if not suggestions:
            suggestions.append("✅ 数据质量优秀，无需优化")

        return suggestions

    def evaluate_batch(self, test_cases, verbose=False):
        """
        批量评估多个测试问题

        Args:
            test_cases: 测试用例列表 [{"question": "...", "keywords": [...]}, ...]
            verbose: 是否显示详细信息

        Returns:
            evaluations: 评估结果列表
        """
        print("\n" + "=" * 80)
        print("🔬 向量库数据质量评估")
        print("=" * 80)
        print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试用例数: {len(test_cases)}")
        print(f"返回结果数: Top-{self.top_k}")
        print(f"向量库路径: {self.config.VECTOR_STORE_PATH}")
        print(f"嵌入模型: {self.config.EMBEDDING_MODEL_PATH}")
        print("=" * 80)

        evaluations = []

        for i, test_case in enumerate(test_cases, 1):
            question = test_case.get('question', '')
            keywords = test_case.get('keywords', [])

            print(f"\n{'='*20} 测试用例 {i}/{len(test_cases)} {'='*20}")

            evaluation = self.evaluate_query(question, keywords, verbose)
            evaluations.append(evaluation)

        # 生成总体评估
        self._generate_summary(evaluations)

        return evaluations

    def _generate_summary(self, evaluations):
        """生成总体评估摘要"""
        print("\n" + "=" * 80)
        print("📋 总体评估摘要")
        print("=" * 80)

        # 计算平均分
        avg_quality_score = sum(e['quality_score'] for e in evaluations) / len(evaluations)
        avg_max_sim = sum(e['max_similarity'] for e in evaluations) / len(evaluations)
        avg_avg_sim = sum(e['avg_similarity'] for e in evaluations) / len(evaluations)

        print(f"\n📊 整体统计:")
        print(f"  - 测试用例数: {len(evaluations)}")
        print(f"  - 平均质量得分: {avg_quality_score:.2f}/100")
        print(f"  - 平均最高相似度: {avg_max_sim:.4f}")
        print(f"  - 平均平均相似度: {avg_avg_sim:.4f}")

        # 质量分级统计
        grades = [e['quality_grade'] for e in evaluations]
        print(f"\n⭐ 质量分布:")
        grade_counts = {}
        for grade in grades:
            grade_key = grade.split()[0]
            grade_counts[grade_key] = grade_counts.get(grade_key, 0) + 1

        for grade, count in sorted(grade_counts.items(), reverse=True):
            print(f"  - {grade}: {count} 个用例")

        # 推荐阈值
        all_top3_sims = []
        all_top5_sims = []
        for e in evaluations:
            sims = [r['similarity'] for r in e['results']]
            all_top3_sims.append(sum(sims[:3])/3)
            all_top5_sims.append(sum(sims[:5])/5)

        global_top3_avg = sum(all_top3_sims) / len(all_top3_sims)
        global_top5_avg = sum(all_top5_sims) / len(all_top5_sims)

        print(f"\n💡 全局阈值建议:")
        print(f"  - 严格模式: {max(0.3, global_top3_avg - 0.05):.2f} (适合高质量回答)")
        print(f"  - 平衡模式: {max(0.25, global_top5_avg - 0.08):.2f} (推荐)")
        print(f"  - 宽松模式: {max(0.2, global_top5_avg - 0.15):.2f} (适合高召回率)")

        # 总体建议
        print(f"\n🔧 总体建议:")
        if avg_quality_score >= 70:
            print("  ✅ 向量库质量优秀，可以直接使用")
            print(f"  💡 建议配置: SEARCH_TYPE='similarity', RETRIEVER_K=4")
        elif avg_quality_score >= 50:
            print("  ⚠️ 向量库质量良好，但有优化空间")
            print("  💡 建议: 清理文本数据后重建向量库")
        else:
            print("  ❌ 向量库质量较差，强烈建议优化")
            print("  💡 建议:")
            print("     1. 使用 tool/clean_text.py 清洗文本")
            print("     2. 调整 CHUNK_SIZE 参数（尝试400-600）")
            print("     3. 重建向量库")

        print("=" * 80)




def save_report(evaluations, output_file):
    """保存评估报告到文件"""
    print(f"\n💾 保存评估报告到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("向量库数据质量评估报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试用例数: {len(evaluations)}\n")
        f.write("\n")

        # 详细结果
        for i, evaluation in enumerate(evaluations, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"测试用例 {i}: {evaluation['question']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"质量得分: {evaluation['quality_score']:.2f}/100\n")
            f.write(f"质量等级: {evaluation['quality_grade']}\n")
            f.write(f"最高相似度: {evaluation['max_similarity']:.4f}\n")
            f.write(f"平均相似度: {evaluation['avg_similarity']:.4f}\n")
            f.write("\n")

            # Top-5结果
            f.write("Top-5 检索结果:\n")
            f.write("-" * 80 + "\n")
            for result in evaluation['results'][:5]:
                f.write(f"第{result['rank']}名 | 相似度: {result['similarity']:.4f} | "
                       f"来源: {os.path.basename(result['source'])}\n")
                f.write(f"内容: {result['content'][:150]}...\n")
                f.write("-" * 80 + "\n")

        # 总体摘要
        avg_quality = sum(e['quality_score'] for e in evaluations) / len(evaluations)
        f.write(f"\n{'='*80}\n")
        f.write("总体评估\n")
        f.write(f"{'='*80}\n")
        f.write(f"平均质量得分: {avg_quality:.2f}/100\n")

    print(f"✅ 报告已保存")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='向量库数据质量评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--top-k', type=int, default=10,
                        help='返回前N个结果（默认：10）')
    parser.add_argument('--output', metavar='FILE',
                        help='保存评估报告到文件')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')

    args = parser.parse_args()

    # 使用内置测试用例
    print(f"📋 使用 {len(TEST_CASES)} 个内置测试用例")
    print("💡 要修改测试用例，请编辑本文件顶部的 TEST_CASES 列表\n")

    # 创建评估器
    evaluator = VectorStoreEvaluator(config, top_k=args.top_k)

    # 执行评估
    evaluations = evaluator.evaluate_batch(TEST_CASES, verbose=args.verbose)

    # 保存报告
    if args.output:
        save_report(evaluations, args.output)

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()

