"""
文本数据清洗工具
用于清理TXT文件中的广告、多余空格、换行等无用内容

使用方法：
    python tool/clean_text.py [选项]

选项：
    --input DIR         输入目录（默认：./data）
    --output DIR        输出目录（默认：./data_cleaned）
    --preview FILE      预览模式：只显示清洗效果，不保存
    --inplace           就地修改（覆盖原文件，危险！）
    --min-length N      最小文本块长度（默认：50）
    --help              显示帮助信息

示例：
    # 清洗data目录的所有TXT文件
    python tool/clean_text.py

    # 预览单个文件的清洗效果
    python tool/clean_text.py --preview data/01.养狗宝典.txt

    # 就地修改（覆盖原文件）
    python tool/clean_text.py --inplace
"""

import sys
import io
import os
import re
import argparse
from pathlib import Path

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ========== 清洗配置 ==========

# 广告和垃圾关键词（参考 config.py）
SPAM_KEYWORDS = [
    # 广告相关
    'pinmuch', '品品品', '网购返利',
    # 盗版水印
    '获取本书PDF全文', '请前往', '搜索该书名',
    '欢速药夯患PDF', '全文，请前往',
    # 网址链接
    'http://', 'https://', 'www.',
    # 推广内容
    '精品资源', '扫码关注', '微信公众号', '公众号',
    '尽在""', '献给', '专业关爱',
    # 版权声明
    '版权所有', '保留所有权利',
    # 其他无用内容
    '流产康复', '做个爱她的男人', '做个体己的女子',
]

# 最小有用文本长度（字符数）
MIN_USEFUL_LENGTH = 50


class TextCleaner:
    """文本清洗器"""

    def __init__(self, spam_keywords=None, min_length=None):
        """
        初始化清洗器

        Args:
            spam_keywords: 垃圾关键词列表
            min_length: 最小有用文本长度
        """
        self.spam_keywords = spam_keywords or SPAM_KEYWORDS
        self.min_length = min_length or MIN_USEFUL_LENGTH

        self.stats = {
            'total_lines': 0,
            'removed_lines': 0,
            'spam_filtered': 0,
            'empty_filtered': 0,
            'short_filtered': 0
        }

    def contains_spam(self, text):
        """检查文本是否包含垃圾关键词"""
        text_lower = text.lower()
        for keyword in self.spam_keywords:
            if keyword.lower() in text_lower:
                return True, keyword
        return False, None

    def clean_line(self, line):
        """清洗单行文本"""
        # 移除两端空格
        line = line.strip()

        # 移除多余空格（连续空格转为单个）
        line = re.sub(r'\s+', ' ', line)

        return line

    def clean_text(self, text):
        """
        清洗文本内容

        Args:
            text: 原始文本

        Returns:
            cleaned_text: 清洗后的文本
            stats: 清洗统计信息
        """
        lines = text.split('\n')
        cleaned_lines = []

        self.stats['total_lines'] = len(lines)

        for line in lines:
            # 清洗行
            cleaned_line = self.clean_line(line)

            # 过滤空行
            if not cleaned_line:
                self.stats['empty_filtered'] += 1
                self.stats['removed_lines'] += 1
                continue

            # 过滤垃圾内容
            is_spam, keyword = self.contains_spam(cleaned_line)
            if is_spam:
                self.stats['spam_filtered'] += 1
                self.stats['removed_lines'] += 1
                continue

            # 过滤过短文本（可能是页码、标题等）
            if len(cleaned_line) < self.min_length:
                self.stats['short_filtered'] += 1
                self.stats['removed_lines'] += 1
                continue

            cleaned_lines.append(cleaned_line)

        # 合并行，用双换行分隔（保持段落结构）
        cleaned_text = '\n\n'.join(cleaned_lines)

        # 移除3个以上连续换行
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text, self.stats.copy()

    def clean_file(self, input_path, output_path=None):
        """
        清洗单个文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（None则不保存）

        Returns:
            cleaned_text: 清洗后的文本
            stats: 统计信息
        """
        # 重置统计
        self.stats = {
            'total_lines': 0,
            'removed_lines': 0,
            'spam_filtered': 0,
            'empty_filtered': 0,
            'short_filtered': 0
        }

        # 读取原文件
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 清洗文本
        cleaned_text, stats = self.clean_text(text)

        # 保存文件（如果指定了输出路径）
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

        return cleaned_text, stats


def preview_file(file_path, cleaner):
    """预览文件清洗效果"""
    print("=" * 80)
    print(f"📄 文件: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    # 清洗文件（不保存）
    cleaned_text, stats = cleaner.clean_file(file_path)

    # 显示统计
    print("\n📊 清洗统计:")
    print(f"  - 总行数: {stats['total_lines']}")
    print(f"  - 移除行数: {stats['removed_lines']} ({stats['removed_lines']/stats['total_lines']*100:.1f}%)")
    print(f"    • 空行: {stats['empty_filtered']}")
    print(f"    • 垃圾内容: {stats['spam_filtered']}")
    print(f"    • 过短文本: {stats['short_filtered']}")
    print(f"  - 保留行数: {stats['total_lines'] - stats['removed_lines']}")

    # 显示前500字符预览
    print("\n📝 清洗后预览（前500字符）:")
    print("-" * 80)
    print(cleaned_text[:500])
    if len(cleaned_text) > 500:
        print("\n... （省略剩余 {} 字符）".format(len(cleaned_text) - 500))
    print("-" * 80)

    print(f"\n💾 清洗后文本长度: {len(cleaned_text)} 字符")
    print("=" * 80)


def clean_directory(input_dir, output_dir, cleaner, inplace=False):
    """清洗目录中的所有TXT文件"""
    print("=" * 80)
    print("🧹 批量清洗TXT文件")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir if not inplace else '原地修改'}")
    print(f"最小文本长度: {cleaner.min_length} 字符")
    print(f"垃圾关键词数: {len(cleaner.spam_keywords)} 个")
    print("=" * 80)

    # 查找所有TXT文件
    txt_files = list(Path(input_dir).glob('**/*.txt'))

    if not txt_files:
        print(f"\n❌ 未找到TXT文件: {input_dir}")
        return

    print(f"\n找到 {len(txt_files)} 个TXT文件\n")

    total_stats = {
        'files_processed': 0,
        'total_lines': 0,
        'removed_lines': 0,
        'spam_filtered': 0,
        'empty_filtered': 0,
        'short_filtered': 0
    }

    # 处理每个文件
    for i, txt_file in enumerate(txt_files, 1):
        file_name = txt_file.name
        print(f"[{i}/{len(txt_files)}] 清洗: {file_name}...")

        # 确定输出路径
        if inplace:
            output_path = str(txt_file)
        else:
            relative_path = txt_file.relative_to(input_dir)
            output_path = os.path.join(output_dir, relative_path)

        try:
            # 清洗文件
            cleaned_text, stats = cleaner.clean_file(str(txt_file), output_path)

            # 累计统计
            total_stats['files_processed'] += 1
            total_stats['total_lines'] += stats['total_lines']
            total_stats['removed_lines'] += stats['removed_lines']
            total_stats['spam_filtered'] += stats['spam_filtered']
            total_stats['empty_filtered'] += stats['empty_filtered']
            total_stats['short_filtered'] += stats['short_filtered']

            # 显示进度
            print(f"  ✅ 完成: 移除 {stats['removed_lines']}/{stats['total_lines']} 行 "
                  f"({stats['removed_lines']/stats['total_lines']*100:.1f}%)")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    # 显示总体统计
    print("\n" + "=" * 80)
    print("📊 清洗总结")
    print("=" * 80)
    print(f"处理文件数: {total_stats['files_processed']}")
    print(f"总行数: {total_stats['total_lines']}")
    print(f"移除行数: {total_stats['removed_lines']} ({total_stats['removed_lines']/total_stats['total_lines']*100:.1f}%)")
    print(f"  - 空行: {total_stats['empty_filtered']}")
    print(f"  - 垃圾内容: {total_stats['spam_filtered']}")
    print(f"  - 过短文本: {total_stats['short_filtered']}")
    print(f"保留行数: {total_stats['total_lines'] - total_stats['removed_lines']}")
    print("=" * 80)

    if not inplace:
        print(f"\n✅ 清洗后的文件已保存到: {output_dir}")
        print("💡 提示: 请检查清洗效果，确认无误后可复制到data目录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='文本数据清洗工具 - 去除广告、多余空格和换行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 清洗data目录的所有TXT文件到data_cleaned目录
    python tool/clean_text.py

    # 预览单个文件的清洗效果
    python tool/clean_text.py --preview data/01.养狗宝典.txt

    # 自定义输入输出目录
    python tool/clean_text.py --input ./raw_data --output ./cleaned_data

    # 就地修改（覆盖原文件，请谨慎使用）
    python tool/clean_text.py --inplace --input ./data

    # 自定义最小文本长度
    python tool/clean_text.py --min-length 100
        """
    )

    parser.add_argument('--input', default='./data',
                        help='输入目录（默认：./data）')
    parser.add_argument('--output', default='./data_cleaned',
                        help='输出目录（默认：./data_cleaned）')
    parser.add_argument('--preview', metavar='FILE',
                        help='预览模式：只显示清洗效果，不保存')
    parser.add_argument('--inplace', action='store_true',
                        help='就地修改（覆盖原文件，危险！）')
    parser.add_argument('--min-length', type=int, default=50,
                        help='最小文本块长度（默认：50）')

    args = parser.parse_args()

    # 创建清洗器
    cleaner = TextCleaner(
        spam_keywords=SPAM_KEYWORDS,
        min_length=args.min_length
    )

    # 预览模式
    if args.preview:
        preview_file(args.preview, cleaner)
        return

    # 就地修改警告
    if args.inplace:
        print("\n⚠️  警告：你选择了 --inplace 模式，这将覆盖原文件！")
        response = input("确认继续？(yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("已取消操作")
            return

    # 批量清洗
    clean_directory(
        input_dir=args.input,
        output_dir=args.output if not args.inplace else args.input,
        cleaner=cleaner,
        inplace=args.inplace
    )


if __name__ == "__main__":
    main()

