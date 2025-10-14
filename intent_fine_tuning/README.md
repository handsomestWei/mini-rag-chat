# 意图识别微调

基于 m3e-small 的轻量级意图分类器，用于识别用户查询的意图，优化RAG系统的检索策略。

## 🎯 功能特性

- **轻量级**: 基于现有的 m3e-small 模型，只需训练小的分类头
- **多语言**: 支持中英文混合查询
- **高准确率**: 在简单任务上达到较高的分类准确率
- **可配置**: 支持自定义意图类别和响应策略（通过 `fine_tuning_config.py`）
- **数据分离**: 通用数据和领域数据分开管理，便于领域适配

## 📋 意图类别

| 类别 | 说明 | 示例 | RAG策略 |
|------|------|------|---------|
| `greeting` | 问候/自我介绍 | "你好", "你是谁", "你能做什么" | 跳过RAG，直接回复 |
| `politeness` | 礼貌用语 | "谢谢", "再见", "不客气" | 跳过RAG，直接回复 |
| `system` | 系统查询 | "系统提示词", "什么模型", "技术栈" | 跳过RAG，安全回复 |
| `chitchat` | 闲聊 | "现在几点", "天气怎么样", "讲个笑话" | 跳过RAG，友好拒绝 |
| `knowledge` | 知识问答 | "狗狗需要每天遛吗", "如何训练狗狗" | 正常RAG流程 |
| `unknown` | 未知/其他 | "嗯", "啊", "哦" | 默认RAG流程 |

## ⚙️ 配置文件

本项目使用 `fine_tuning_config.py` 集中管理所有配置，便于统一修改和维护。

### 配置文件结构

```python
# 模型路径配置
DEFAULT_EMBEDDING_MODEL = "../model/moka-ai/m3e-small"
DEFAULT_SAVE_DIR = "./model/intent-classifier"

# 数据文件配置
DEFAULT_GENERAL_DATA = "training_data_general.json"
DEFAULT_DOMAIN_DATA = "training_data_domain.json"
DEFAULT_TEST_DATA = "test_data.json"

# 训练参数配置
DEFAULT_TEST_SIZE = 0.2
CLASSIFIER_PARAMS = {...}

# 意图类别定义
INTENT_CLASSES = ["greeting", "knowledge", ...]

# 意图响应配置
INTENT_CONFIG = {...}
```

### 修改配置的好处

1. **一次修改，全局生效**：所有脚本自动使用新配置
2. **集中管理**：所有参数在一个文件中，易于查找和维护
3. **避免硬编码**：不需要在多个文件中重复修改
4. **便于版本控制**：配置变更有清晰的历史记录

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. (可选) 修改配置

如需自定义模型、意图类别或响应，编辑 `fine_tuning_config.py`：

```python
# 例如：更换embedding模型
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 例如：修改意图回复
INTENT_CONFIG["greeting"]["response"] = "欢迎使用XXX系统！"
```

### 3. 训练模型

```bash
# 使用默认参数训练
python train.py

# 自定义参数（会覆盖配置文件）
python train.py --model_path ../model/moka-ai/m3e-small --save_dir ./model/intent-classifier --test_size 0.2
```

### 4. 复制模型到主目录

训练完成后，需要将模型复制到主项目的模型目录才能使用

### 5. 测试模型

```bash
# 测试所有预设用例
python test.py

# 测试特定文本
python test.py --text "你好"
```

### 6. 在代码中使用

```python
from intent_classifier import IntentClassifier

# 创建分类器
classifier = IntentClassifier(model_path="../model/moka-ai/m3e-small")

# 加载训练好的模型
classifier.load_model("./model/intent-classifier")

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
- **用途**: 文本向量化，将文本转换为512维向量
- **配置位置**: `intent_classifier.py` 中的 `DEFAULT_EMBEDDING_MODEL`

### 微调模型路径
- **训练输出**: `./model/intent-classifier/` (训练脚本的输出目录)
- **实际使用**: `../model/intent-classifier/` (主项目的模型目录)
- **包含文件**:
  - `intent_classifier.pkl`: 训练好的LogisticRegression分类器
  - `label_encoder.json`: 标签编码和配置信息
- **用途**: 意图分类，判断用户输入的意图类别
- **配置位置**: `intent_classifier.py` 中的 `DEFAULT_SAVE_DIR`
- **注意**: 训练输出在 `./model/intent-classifier/`，使用时复制到 `../model/intent-classifier/`

### 🔧 修改模型路径

**方式1：修改配置文件**（推荐，一次修改全局生效）

编辑 `fine_tuning_config.py` 中的配置：
```python
# ========== 模型路径配置 ==========
# 预训练embedding模型路径
DEFAULT_EMBEDDING_MODEL = "../model/moka-ai/m3e-small"
# 或换成其他模型：
# DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# DEFAULT_EMBEDDING_MODEL = "../model/moka-ai/m3e-base"

# 训练输出目录
DEFAULT_SAVE_DIR = "./model/intent-classifier"

# 训练数据文件
DEFAULT_GENERAL_DATA = "training_data_general.json"
DEFAULT_DOMAIN_DATA = "training_data_domain.json"
DEFAULT_TEST_DATA = "test_data.json"

# 训练参数
DEFAULT_TEST_SIZE = 0.2
CLASSIFIER_PARAMS = {
    "random_state": 42,
    "max_iter": 1000,
    "multi_class": "ovr"
}
```

**方式2：命令行参数**（临时修改）

```bash
# 使用不同的embedding模型训练
python train.py --model_path BAAI/bge-small-zh-v1.5

# 保存到不同目录
python train.py --save_dir ./model/my-classifier

# 修改测试集比例
python train.py --test_size 0.3
```

## 🔧 配置说明

### 意图配置

所有配置集中在 `fine_tuning_config.py` 文件中，修改后所有脚本自动生效。

#### 修改意图回复

编辑 `fine_tuning_config.py` 中的 `INTENT_CONFIG`：

```python
INTENT_CONFIG = {
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

1. 在 `fine_tuning_config.py` 的 `INTENT_CLASSES` 中添加新类别
2. 在 `INTENT_CONFIG` 中添加该类别的配置
3. 在 `training_data_general.json` 或 `training_data_domain.json` 中添加训练样本
4. 重新训练模型：`python train.py`

## 📁 文件结构

```
intent_fine_tuning/
├── fine_tuning_config.py        # 配置文件 ⭐ 集中管理所有配置
├── intent_classifier.py          # 主分类器代码
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── training_data_general.json   # 通用训练数据（可复用）
├── training_data_domain.json    # 领域训练数据（需修改）
├── test_data.json               # 测试数据文件
├── requirements.txt             # 依赖包
├── README.md                   # 说明文档
├── 意图识别集成指南.md          # 集成指南
└── model/intent-classifier/     # 训练好的模型 (训练后生成)
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

在 `fine_tuning_config.py` 中修改 `CLASSIFIER_PARAMS`：

```python
CLASSIFIER_PARAMS = {
    "random_state": 42,
    "max_iter": 1000,      # 最大迭代次数
    "C": 1.0,              # 正则化强度（可选）
    "multi_class": "ovr"   # 一对多策略
}
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
