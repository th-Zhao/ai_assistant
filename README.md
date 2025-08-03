# 🎓 智能学习助手系统

基于RAG（检索增强生成）技术的智能学习助手，集成大语言模型和向量检索技术，为用户提供个性化的学习辅助服务。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 功能特性

### 📄 智能文档处理
- **多格式支持**：PDF、DOCX、TXT文档上传解析
- **智能分块**：自动文档切分和向量化存储
- **持久化保存**：文档和向量数据完整持久化，重启不丢失

### 🤖 智能问答系统
- **三种问答模式**：
  - 🧠 **纯LLM模式**：基于模型通用知识
  - 📖 **单文档模式**：基于特定文档内容
  - 📚 **多文档模式**：跨文档知识整合
- **语义理解**：支持自然语言提问
- **上下文记忆**：多轮对话保持连贯性

### 📚 学习辅助功能
- **📝 文档总结**：自动提取核心内容和关键知识点
- **❓ 练习题生成**：智能生成选择题、判断题、简答题
- **📋 学习计划**：个性化学习计划制定
- **💡 概念解释**：智能概念解释和知识扩展

### 🔄 高级特性
- **多模型支持**：OpenAI GPT-4 + DeepSeek备用
- **会话管理**：Redis支持的持久化会话
- **并发处理**：支持多用户同时使用
- **错误恢复**：自动重试和故障转移

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│                 前端界面                      │
│        HTML5 + JavaScript + Bootstrap       │
└─────────────────────┬───────────────────────┘
                      │ REST API
┌─────────────────────┴───────────────────────┐
│                FastAPI后端                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   AI服务    │ │  文档服务   │ │ RAG服务 │ │
│  └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────┴───────────────────────┐
│                 数据存储                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ FAISS向量库 │ │ 文档存储    │ │ Redis   │ │
│  └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────┘
```

## 🚀 快速开始

### 方法一：一键启动（推荐）

#### 使用Conda环境（推荐）
```bash
# 克隆项目
git clone <repository-url>
cd ai_assistant

# 一键启动（自动创建环境、安装依赖、启动服务）
python start_conda.py
```

#### 使用系统Python
```bash
# 克隆项目
git clone <repository-url>
cd intelligent-study-assistant

# 一键启动
python start.py
```

### 方法二：手动部署

#### 1. 环境准备
```bash
# 确保Python 3.8+
python --version

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

#### 2. 安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 或使用清华源加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3. 配置API密钥
在 `backend/config.py` 中配置您的API密钥：
```python
# OpenAI API配置
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# DeepSeek API配置（备用）
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
```

#### 4. 启动Redis（可选，用于会话管理）
```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

#### 5. 启动服务
```bash
# 启动后端
cd backend
python main.py

# 打开前端（新终端）
# 直接用浏览器打开 frontend/index.html
```

### 访问应用
- **前端界面**：打开 `frontend/index.html`
- **后端API**：http://localhost:8000
- **API文档**：http://localhost:8000/docs

## 📖 使用指南

### 1. 文档上传
1. 点击"选择文件"按钮
2. 选择PDF、DOCX或TXT文件（最大50MB）
3. 等待文档解析和向量化完成

### 2. 智能问答
1. 在文档列表中选择要查询的文档
2. 在问答框中输入问题
3. 选择使用的AI模型（OpenAI或DeepSeek）
4. 点击"发送"获取回答

### 3. 功能模块
- **📝 生成总结**：点击后自动分析文档核心内容
- **❓ 生成练习题**：根据文档内容生成测试题目
- **📋 制定学习计划**：个性化学习路径规划
- **💡 概念解释**：输入概念获取详细解释

## 🔧 API接口

### 主要端点

| 端点 | 方法 | 功能 | 参数 |
|------|------|------|------|
| `/upload` | POST | 上传文档 | `file`: 文档文件 |
| `/ask` | POST | 智能问答 | `question`, `document_ids`, `use_deepseek` |
| `/summary` | POST | 生成总结 | `document_ids`, `use_deepseek` |
| `/quiz` | POST | 生成练习题 | `document_ids`, `count`, `use_deepseek` |
| `/study-plan` | POST | 制定学习计划 | `document_ids`, `level`, `use_deepseek` |
| `/explain` | POST | 概念解释 | `concept`, `use_deepseek` |
| `/documents` | GET | 获取文档列表 | 无 |
| `/health` | GET | 健康检查 | 无 |

### 请求示例

```bash
# 上传文档
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# 智能问答
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是机器学习？",
    "document_ids": ["doc1", "doc2"],
    "use_deepseek": false
  }'

# 生成总结
curl -X POST "http://localhost:8000/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1"],
    "use_deepseek": false
  }'
```

## 📁 项目结构

```
intelligent-study-assistant/
├── backend/                    # 后端代码
│   ├── main.py                # FastAPI主程序
│   ├── config.py              # 配置文件
│   └── services/              # 业务逻辑
│       ├── ai_service.py      # AI模型服务
│       ├── document_service.py # 文档处理服务
│       ├── rag_service.py     # RAG检索服务
│       └── session_service.py # 会话管理服务
├── frontend/                   # 前端代码
│   └── index.html             # 主页面
├── uploads/                    # 文件上传目录
├── vector_store/              # 向量数据存储
├── requirements.txt           # Python依赖
├── environment.yml            # Conda环境配置
├── start.py                   # 通用启动脚本
├── start_conda.py            # Conda启动脚本
└── README.md                 # 项目文档
```

## ⚙️ 配置说明

### 模型配置
```python
# backend/config.py
class Config:
    # OpenAI配置
    OPENAI_API_KEY = "your-key"
    OPENAI_MODEL = "gpt-4o-mini"
    
    # DeepSeek配置
    DEEPSEEK_API_KEY = "your-key"
    DEEPSEEK_MODEL = "deepseek-r1"
    
    # RAG配置
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # 向量模型
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 环境变量
```bash
# .env文件（可选）
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

## 🔧 开发指南

### 本地开发
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black backend/
flake8 backend/

# 启动开发服务器
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker部署
```bash
# 构建镜像
docker build -t intelligent-study-assistant .

# 运行容器
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  intelligent-study-assistant
```

## 🐛 常见问题

### Q1: 依赖安装失败
**A:** 尝试使用清华源或升级pip：
```bash
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 文档上传失败
**A:** 检查文件格式和大小：
- 支持格式：PDF、DOCX、TXT
- 文件大小：<50MB
- 确保文件未损坏

### Q3: Redis连接失败
**A:** 确保Redis服务运行：
```bash
# 检查Redis状态
redis-cli ping

# 启动Redis
redis-server
```

### Q4: API调用超时
**A:** 检查网络连接和API密钥：
- 验证API密钥有效性
- 检查网络代理设置
- 确认API额度充足

### Q5: 向量检索慢
**A:** 优化向量存储：
- 减少文档块大小
- 使用更快的嵌入模型
- 考虑使用GPU加速

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. **Fork项目** → 点击右上角Fork按钮
2. **创建分支** → `git checkout -b feature/your-feature`
3. **提交更改** → `git commit -am 'Add some feature'`
4. **推送分支** → `git push origin feature/your-feature`
5. **创建PR** → 在GitHub上创建Pull Request

### 代码规范
- 使用Python PEP 8代码风格
- 添加适当的注释和文档
- 编写单元测试
- 确保所有测试通过

### 问题报告
如果发现bug或有功能建议，请[创建Issue](../../issues)并提供：
- 详细的问题描述
- 复现步骤
- 期望行为
- 系统环境信息

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

感谢以下开源项目：
- [FastAPI](https://fastapi.tiangolo.com/) - 现代高性能Web框架
- [LangChain](https://langchain.readthedocs.io/) - 大语言模型应用框架
- [FAISS](https://faiss.ai/) - 高效向量检索库
- [Bootstrap](https://getbootstrap.com/) - 前端UI框架

## 📞 联系方式

- 📧 邮箱：[your-email@example.com]
- 💬 讨论：[GitHub Discussions](../../discussions)
- 🐛 问题：[GitHub Issues](../../issues)

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！** 