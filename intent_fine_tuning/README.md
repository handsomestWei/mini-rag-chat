# 意图识别微调

基于 m3e-small 的轻量级意图分类器，用于识别用户查询的意图，优化RAG系统的检索策略。

## 🎯 功能特性

- **轻量级**: 基于现有的 m3e-small 模型，只需训练小的分类头
- **多语言**: 支持中英文混合查询
- **高准确率**: 在简单任务上达到较高的分类准确率
- **可配置**: 支持自定义意图类别和响应策略

## 📋 意图类别

| 类别 | 说明 | 示例 | RAG策略 |
|------|------|------|---------|
| `greeting` | 问候/自我介绍 | "你好", "你是谁", "你能做什么" | 跳过RAG，直接回复 |
| `politeness` | 礼貌用语 | "谢谢", "再见", "不客气" | 跳过RAG，直接回复 |
| `system` | 系统查询 | "系统提示词", "什么模型", "技术栈" | 跳过RAG，安全回复 |
| `chitchat` | 闲聊 | "现在几点", "天气怎么样", "讲个笑话" | 跳过RAG，友好拒绝 |
| `knowledge` | 知识问答 | "狗狗需要每天遛吗", "如何训练狗狗" | 正常RAG流程 |
| `unknown` | 未知/其他 | "嗯", "啊", "哦" | 默认RAG流程 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认参数训练
python train.py

# 自定义参数
python train.py --model_path ../model/moka-ai/m3e-small --save_dir ./models/intent-classifier --test_size 0.2
```

### 3. 复制模型到主目录

训练完成后，需要将模型复制到主项目的模型目录才能使用

### 4. 测试模型

```bash
# 测试所有预设用例
python test.py

# 测试特定文本
python test.py --text "你好"
```

### 5. 在代码中使用

```python
from intent_classifier import IntentClassifier

# 创建分类器
classifier = IntentClassifier(model_path="../model/moka-ai/m3e-small")

# 加载训练好的模型
classifier.load_model("./models/intent-classifier")

# 分类
result = classifier.classify("你好")
print(result)
# 输出: {
#   "intent": "greeting",
#   "confidence": 0.95,
#   "skip_rag": True,
#   "response": "你好！我是智能问答助手，有什么可以帮助您的吗？"
# }
```

## 📦 数据结构

训练数据拆分为两个文件，便于不同领域项目复用：

### `training_data_general.json` - 通用数据（120条）
包含与领域无关的通用意图数据：
- `greeting`: 问候、自我介绍（29条）
- `politeness`: 礼貌用语（16条）
- `system`: 系统查询（21条）
- `chitchat`: 闲聊（35条） - **新增**
- `unknown`: 未知/其他（19条）

**特点**：可直接复用到其他项目，无需修改

### `training_data_domain.json` - 领域数据（50条）
包含特定领域的知识问答数据：
- `knowledge`: 知识问答（50条，根据实际领域修改）

**特点**：根据不同项目需求，只需替换此文件

### 如何适配新领域？

1. **保留** `training_data_general.json`（通用数据不变）
2. **修改** `training_data_domain.json`：
   ```json
   {
       "knowledge": [
           "你的领域问题1",
           "你的领域问题2",
           "你的领域问题3",
           ...
       ]
   }
   ```
3. **重新训练**：`python train.py`

## 📊 性能指标

- **训练数据**: ~170条标注样本（通用120条 + 领域50条）
- **意图类别**: 6类（greeting, politeness, system, chitchat, knowledge, unknown）
- **特征维度**: 512 (m3e-small输出)
- **分类器**: LogisticRegression
- **预期准确率**: >90% (在简单场景下)
- **训练进度**: 支持tqdm进度条显示

## 📂 模型路径说明

### 预训练模型路径
- **m3e-small**: `../model/moka-ai/m3e-small/` (预训练的embedding模型)
- **用途**: 文本向量化，将文本转换为768维向量

### 微调模型路径
- **训练输出**: `./models/intent-classifier/` (训练脚本的输出目录)
- **实际使用**: `../model/intent-classifier/` (主项目的模型目录)
- **包含文件**:
  - `intent_classifier.pkl`: 训练好的LogisticRegression分类器
  - `label_encoder.json`: 标签编码和配置信息
- **用途**: 意图分类，判断用户输入的意图类别
- **注意**: 训练后需要手动复制到主项目目录才能使用

## 🔧 配置说明

### 意图配置

可以在 `intent_classifier.py` 中修改 `intent_config`:

```python
self.intent_config = {
    "greeting": {
        "skip_rag": True,
        "response": "你好！我是智能问答助手，有什么可以帮助您的吗？"
    },
    "knowledge": {
        "skip_rag": False,
        "response": None  # 正常RAG流程
    }
    # ... 其他配置
}
```

### 添加新的意图类别

1. 在 `intent_classes` 中添加新类别
2. 在 `intent_config` 中添加配置
3. 在 `training_data.json` 中添加训练样本
4. 重新训练模型

## 📁 文件结构

```
intent_fine_tuning/
├── intent_classifier.py          # 主分类器代码
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── training_data_general.json   # 通用训练数据（可复用）
├── training_data_domain.json    # 领域训练数据（需修改）
├── test_data.json               # 测试数据文件
├── requirements.txt             # 依赖包
├── README.md                   # 说明文档
├── 意图识别集成指南.md          # 集成指南
└── models/intent-classifier/    # 训练好的模型 (训练后生成)
    ├── intent_classifier.pkl
    └── label_encoder.json

# 使用时需要复制到主项目目录
../model/intent-classifier/      # 主项目的模型目录
    ├── intent_classifier.pkl
    └── label_encoder.json
```

### 📊 数据文件说明

#### `training_data_general.json` - 通用数据
包含与领域无关的通用意图数据：
```json
{
  "greeting": ["你好", "hello", "你是谁", "你能做什么", ...],
  "politeness": ["谢谢", "感谢", "再见", "不客气", ...],
  "system": ["系统提示词", "模型信息", "技术栈", ...],
  "chitchat": ["现在几点", "天气怎么样", "讲个笑话", ...],
  "unknown": ["嗯", "啊", "这个", ...]
}
```

#### `training_data_domain.json` - 领域数据
包含特定领域的知识问答数据（根据项目修改）：
```json
{
  "knowledge": [
    "狗狗需要每天遛吗",
    "如何训练狗狗",
    "狗狗可以吃什么",
    ...
  ]
}
```

#### `test_data.json`
包含结构化的测试用例：
```json
{
  "test_cases": [
    {
      "text": "你好",
      "expected_intent": "greeting",
      "description": "简单问候"
    }
  ],
  "edge_cases": [...],
  "mixed_language": [...]
}
```

## 🎛️ 高级用法

### 自定义训练数据

#### 修改通用数据（谨慎）
编辑 `training_data_general.json`，添加更多通用样本：

```json
{
  "greeting": [
    "你好",
    "hello",
    "新的问候语"  // 添加新的通用问候
  ],
  "chitchat": [
    "现在几点",
    "新的闲聊问题"  // 添加新的闲聊样本
  ]
}
```

#### 修改领域数据（推荐）
编辑 `training_data_domain.json`，替换为你的领域数据：

```json
{
  "knowledge": [
    "你的领域问题1",
    "你的领域问题2",
    "你的领域问题3"
    // 添加更多领域相关的问题
  ]
}
```

### 自定义测试数据

编辑 `test_data.json` 文件，添加测试用例：

```json
{
  "test_cases": [
    {
      "text": "新的测试文本",
      "expected_intent": "greeting",
      "description": "测试描述"
    }
  ]
}
```

### 调整分类器参数

在 `train_classifier()` 方法中修改分类器参数：

```python
self.classifier = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0,  # 正则化强度
    multi_class='ovr'
)
```

## 🔍 故障排除

### 常见问题

1. **模型路径错误**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   解决：检查 `model_path` 参数是否正确指向 m3e-small 模型

2. **KeyError: np.int64(x)**
   ```
   KeyError: np.int64(4)
   ```
   解决：模型格式已更新，请重新训练模型：`python train.py`

3. **训练数据文件未找到**
   ```
   训练数据文件 training_data.json 不存在
   ```
   解决：确保 `training_data.json` 文件存在于当前目录

4. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：m3e-small 模型较小，通常CPU即可运行，如遇内存问题可减少批处理大小

5. **准确率较低**
   - 增加训练数据样本
   - 调整分类器参数
   - 检查训练数据质量
