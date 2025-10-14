# Mini RAG Chat

<div align="center">

**为低配置服务器优化的轻量级RAG对话系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ 核心特性

### 🚀 低配置优化
- **2核4G CPU 即可运行**，无需GPU
- 内存占用优化（<2GB）
- CPU推理加速配置
- 智能批处理和缓存
- **并发控制**：保护服务器资源，防止过载
- **频率限制**：防止恶意攻击和脚本滥用

### 🔒 安全防护
- **输入长度限制**：防止过长输入影响性能
- **内容安全过滤**：防止恶意查询和系统信息泄露
- **敏感词检测**：自动识别并阻止技术细节查询
- **智能截断**：超长输入自动截断而非拒绝
- **频率限制**：基于真实IP的多级频率控制（分钟/小时/天）
- **脚本检测**：自动识别和拦截curl、python-requests等脚本工具
- **代理支持**：正确识别Nginx/CDN后的真实IP

### 🎯 智能意图识别（可选）
- **轻量级分类器**：基于m3e-small的意图识别模型
- **优化RAG调用**：简单问候、礼貌用语跳过RAG，节省资源
- **多意图支持**：问候、知识问答、礼貌用语、系统查询、未知
- **易于训练**：90条样本即可达到>90%准确率
- **详见**：[intent_fine_tuning/README.md](intent_fine_tuning/README.md)

### 🇨🇳 中文优化
- **m3e-small 嵌入模型**：专为中文RAG优化（200MB）
- **Qwen2-1.5B LLM**：强大的中文理解能力（1.5GB）
- 中文分词和语义优化
- 支持中文文档处理

### 📦 模块化设计
- `RAGManager`：文档加载、切分、向量库管理
- `QueryExpander`：查询改写和扩展
- `ChatHandler`：对话处理和上下文管理
- `SecurityFilter`：安全过滤和输入验证
- `DocumentCompressor`：文档压缩和摘要提取
- `IntentClassifier`：意图识别和路由
- `RateLimiter`：频率限制和脚本检测
- 易于扩展和自定义

### 🔄 增量加载
- **无需重建整个向量库**
- 支持 PDF 和 TXT 格式
- 自动文档迁移
- 热更新支持（运行时加载新文档）

### ⚡ 流式输出
- **实时响应**：逐字显示AI回答，首字响应时间缩短80%
- **状态提示**：实时显示检索和生成进度
- **用户体验优化**：无需等待完整生成
- **智能降级**：自动兼容非流式模式

### 🗜️ 文档压缩
- **混合压缩**：TextRank初筛 + m3e语义重排序
- **智能提取**：基于语义相似度选择最相关句子
- **高压缩率**：75%+的压缩率，保留核心信息
- **零额外成本**：复用m3e模型，无需新模型
- **显著提速**：LLM处理速度提升50%+
- **详见**：[WIKI.md - 文档压缩优化](WIKI.md#-文档压缩优化)

### 🛠️ 完整工具集
- **数据清洗工具**（`clean_text.py`）：文本去噪、格式化
- **质量评估工具**（`evaluate_quality.py`）：评估RAG效果
- 完整的日志和监控

---

## 💻 系统要求

### 最低配置
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 10GB 可用空间
- **操作系统**: Windows / Linux / macOS
- **Python**: 3.8+

### 推荐配置
- **CPU**: 4核心
- **内存**: 8GB RAM
- **存储**: 20GB SSD

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/yourusername/mini-rag-chat.git
cd mini-rag-chat
```

### 2️⃣ 安装依赖

```bash
# 官方源（国外）
pip install -r requirements.txt

# 国内用户 - 使用镜像源加速⭐推荐
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**永久配置镜像源**（推荐）：

```bash
# Linux/macOS
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com

# Windows (PowerShell)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
```

### 3️⃣ 下载模型

详见 [WIKI.md](WIKI.md#-模型下载) 章节

### 4️⃣ 准备意图分类器（可选）

如需使用智能意图识别功能优化RAG性能：

```bash
# 进入意图分类器目录
cd intent_fine_tuning

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py

# 测试模型
python test.py

# 返回项目根目录
cd ..
```

**重要**：训练后的模型会保存在 `intent_fine_tuning/model/intent-classifier/`，使用时需要手动拷贝到 `model/intent-classifier/`：

```bash
# Windows (PowerShell)
Copy-Item -Path "intent_fine_tuning\model\intent-classifier\*" -Destination "model\intent-classifier\" -Recurse -Force

# Linux/macOS
cp -r intent_fine_tuning/model/intent-classifier/* model/intent-classifier/
```

详细说明见：[intent_fine_tuning/README.md](intent_fine_tuning/README.md)

### 5️⃣ 准备数据

将你的文档（PDF或TXT）放入 `data/` 目录：

```bash
data/
  ├── document1.txt
  ├── document2.pdf
  └── document3.txt
```

### 6️⃣ 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动

### 7️⃣ 开始对话

在浏览器打开 `http://localhost:5000`，即可开始对话！

---

## 📚 详细文档

📖 **完整使用指南**：[WIKI.md](WIKI.md)

包含以下详细内容：
- 📥 模型下载指南（含国内镜像）
- ⚙️ 详细配置说明
- 📚 使用指南和最佳实践
- 🛠️ 工具集使用说明
- 🌐 API接口文档
- ❓ 常见问题解答
- 📊 性能基准测试
- 🔧 开发指南

📖 **意图分类器文档**：[intent_fine_tuning/README.md](intent_fine_tuning/README.md)

包含以下详细内容：
- 🎯 意图分类器功能特性
- 🚀 快速开始和训练指南
- 📊 性能指标和模型说明
- 🔧 自定义训练数据
- 🔍 故障排除和常见问题


---

## 🌟 项目亮点

### 🎯 专为低配优化
- 2核4G服务器即可流畅运行
- 内存占用控制在2GB以内
- CPU推理优化，无需GPU

### 🔐 企业级安全
- 多层安全防护机制
- 防止系统信息泄露
- 智能输入验证和过滤

### 🚀 极致性能
- 流式输出，首字响应时间<1秒
- 并发控制，保护服务器稳定
- 增量加载，无需重建向量库
- 文档压缩，LLM速度提升50%+
- 智能意图识别，83%查询跳过RAG

### 🇨🇳 中文友好
- 专为中文优化的模型选择
- 完整的中文文档和错误提示
- 国内镜像源加速下载
