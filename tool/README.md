# 🛠️ 工具集说明

本目录包含独立的工具脚本，可直接运行，无需启动主应用。

---

## 📦 工具列表

### 1. 📄 `clean_text.py` - 文本数据清洗工具

**功能**：清理TXT文件中的广告、多余空格、换行等无用内容

**快速使用**：
```powershell
# 清洗 data/ 目录所有TXT到 data_cleaned/
python tool/clean_text.py

# 预览单个文件的清洗效果
python tool/clean_text.py --preview data/01.养狗宝典.txt

# 就地修改（覆盖原文件，请谨慎）
python tool/clean_text.py --inplace
```

**主要参数**：
- `--input DIR` - 输入目录（默认：./data）
- `--output DIR` - 输出目录（默认：./data_cleaned）
- `--preview FILE` - 预览模式，不保存
- `--inplace` - 就地修改（覆盖原文件）
- `--min-length N` - 最小文本长度（默认：50）

**清洗规则**：
- ✅ 过滤广告关键词（pinmuch、精品资源等14个）
- ✅ 删除空行和多余空格
- ✅ 过滤过短文本（<50字符）
- ✅ 规范化格式（段落间双换行）

**适用场景**：
- OCR识别后的文本清洗
- 网络爬取的文本预处理
- 向量库建立前的数据准备

---

### 2. 🔬 `evaluate_quality.py` - 向量库质量评估工具

**功能**：评估向量化质量，为相似度阈值设定提供参考

**快速使用**：
```powershell
# 运行默认评估（8个测试用例）
python tool/evaluate_quality.py

# 返回前20个结果
python tool/evaluate_quality.py --top-k 20

# 保存评估报告
python tool/evaluate_quality.py --output report.txt

# 显示详细信息
python tool/evaluate_quality.py --verbose
```

**主要参数**：
- `--top-k N` - 返回前N个结果（默认：10）
- `--output FILE` - 保存评估报告到文件
- `--verbose` - 显示详细信息（包括完整文档内容）

**评估内容**：
- ✅ 相似度分数分布（最高、平均、Top-3、Top-5）
- ✅ 关键词命中分析（排名、命中率）
- ✅ 质量评级（0-100分，五星等级）
- ✅ 阈值建议（严格、平衡、宽松三种模式）
- ✅ 改进建议（针对性优化建议）

**输出示例**：
```
【第 1 名】
  🎯 L2距离: 0.8234
  📈 相似度: 0.5483
  ✅ 命中关键词: 心丝虫, 传染
  📄 内容: 心丝虫的传染途径...

📊 评估分析
  - 最高相似度: 0.5483
  - 平均相似度: 0.4521
  - 综合得分: 65.50/100
  - 质量等级: 良好 ⭐⭐⭐⭐

💡 阈值建议:
  - 严格模式: 0.49
  - 平衡模式: 0.38
  - 宽松模式: 0.28
```

**自定义测试用例**：
编辑 `evaluate_quality.py` 顶部的 `TEST_CASES` 列表：
```python
TEST_CASES = [
    {
        "question": "你的问题",
        "keywords": ["期望", "关键词"]
    },
    # 添加更多测试用例...
]
```

**适用场景**：
- 评估向量库数据质量
- 为阈值设定提供数据支持
- 验证优化效果（清洗前后对比）
- 诊断检索不准确问题

---

## 🚀 典型工作流

### 工作流1：数据清洗 + 质量评估

```powershell
# 1. 评估当前向量库质量
python tool/evaluate_quality.py

# 2. 如果质量较差，清洗文本数据
python tool/clean_text.py

# 3. 复制清洗后的文件
Copy-Item data_cleaned/* data/ -Force

# 4. 重建向量库
Remove-Item -Recurse -Force ./vector_store
python app_new.py

# 5. 重新评估质量
python tool/evaluate_quality.py

# 6. 对比改善效果
```

### 工作流2：快速质量检查

```powershell
# 快速评估并保存报告
python tool/evaluate_quality.py --output quality_report.txt

# 查看报告
cat quality_report.txt
```

### 工作流3：深度诊断

```powershell
# 详细模式 + 更多结果 + 保存报告
python tool/evaluate_quality.py --top-k 20 --verbose --output detailed_report.txt
```

---

## 📊 工具对比

| 工具 | 功能 | 输入 | 输出 | 用途 |
|------|------|------|------|------|
| `clean_text.py` | 清洗文本 | TXT文件 | 清洗后TXT | 数据预处理 |
| `evaluate_quality.py` | 评估质量 | 向量库 | 评估报告 | 质量诊断 |

---

## 💡 最佳实践

### 1. 定期评估

每次添加新文档后都运行评估：
```powershell
python tool/evaluate_quality.py
```

### 2. 清洗前后对比

```powershell
# 清洗前评估
python tool/evaluate_quality.py --output before.txt

# 清洗并重建
python tool/clean_text.py
# ...重建向量库...

# 清洗后评估
python tool/evaluate_quality.py --output after.txt

# 对比报告
diff before.txt after.txt
```

### 3. 根据评估结果优化

**质量得分 < 50**：
1. 使用 `clean_text.py` 清洗数据
2. 调整 `CHUNK_SIZE` 参数
3. 重建向量库

**质量得分 50-70**：
1. 清理数据或调整参数
2. 考虑更换更好的embedding模型

**质量得分 > 70**：
1. 当前配置已优秀
2. 可以根据阈值建议配置检索参数

---

## ⚙️ 配置说明

### clean_text.py 配置

在文件顶部修改：
```python
SPAM_KEYWORDS = [
    'pinmuch', '品品品',  # 添加你的关键词
    # ...
]

MIN_USEFUL_LENGTH = 50  # 调整最小长度
```

### evaluate_quality.py 配置

在文件顶部修改：
```python
TEST_CASES = [
    {
        "question": "你的问题",
        "keywords": ["关键词1", "关键词2"]
    },
    # 添加更多测试用例
]
```

---

## 🆘 常见问题

### Q1: 清洗工具过滤太严格？

**A**: 调整参数：
```powershell
python tool/clean_text.py --min-length 30  # 降低最小长度
```

或编辑文件，删除不需要的 `SPAM_KEYWORDS`。

### Q2: 评估工具说向量库不存在？

**A**: 先运行应用创建向量库：
```powershell
python app_new.py
# 等待启动完成后，按 Ctrl+C 停止
# 然后再运行评估工具
```

### Q3: 评估结果质量很差怎么办？

**A**: 按照工作流1操作：
1. 清洗文本数据
2. 重建向量库
3. 重新评估

### Q4: 如何添加自己的测试问题？

**A**: 编辑 `evaluate_quality.py`，在 `TEST_CASES` 列表中添加：
```python
TEST_CASES = [
    # ... 现有测试用例 ...
    {
        "question": "你的新问题",
        "keywords": ["关键词1", "关键词2"]
    },
]
```

---

**工具集已就绪！开始优化你的RAG系统吧！** 🚀

