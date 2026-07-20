# PDF 知识库问答系统

基于 LangChain + RAG 的本地 PDF 知识库问答系统。上传 PDF → 自动索引 → 智能问答 → 展示引用来源。

## ✨ 功能特性

- 📄 **PDF 上传**：支持批量上传多个 PDF 文件
- 🔍 **RAG 检索**：基于语义相似度检索最相关的文档段落
- 💬 **智能问答**：结合文档上下文进行精准回答，并展示引用来源
- 💾 **持久化索引**：索引完成后自动保存，下次启动秒级恢复
- 🗑️ **一键清空**：支持清空索引重建知识库
- 📊 **RAG 评测**：内置量化评估体系，可生成评测报告

## 技术栈

| 技术 | 用途 |
|------|------|
| Streamlit | Web UI 框架 |
| LangChain | LLM 编排框架 |
| FAISS | 向量相似度搜索 |
| BGE-small-zh-v1.5 | 中文 Embedding 模型（本地运行） |
| DeepSeek API | 大语言模型 |
| Ragas | RAG 评测指标（忠实度/检索精度） |

## 📊 新增功能：RAG 评测体系

### 两种评估方式
1. **Ragas 专业指标**：Faithfulness（回答忠实度）、Context Precision（检索精度）
2. **自建评估**：关键词命中率、检索相关率（无需 LLM，稳定运行）

### 使用方法
- **界面入口**：侧边栏点击 "📊 运行RAG评测"
- **命令行**：
  ```bash
  python 05_rag_evaluation.py
  ```
  自动运行测试集，生成 `evaluation_report.md` 详细报告

## 快速开始

### 前置条件

- Python >= 3.9
- [DeepSeek API Key](https://platform.deepseek.com)

### 安装

```bash
# 克隆仓库
git clone https://github.com/Huihuihui9/pdf-qa-system.git
cd pdf-qa-system

# 安装依赖（包含 ragas/datasets）
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 DEEPSEEK_API_KEY
```

### 启动

```bash
streamlit run app.py
```

浏览器自动打开 http://localhost:8501，按以下步骤使用：

1. 左侧上传 PDF 文件
2. 点击「开始索引」
3. 在对话框输入问题
4. 查看回答和引用来源

## 项目结构

```
pdf-qa-system/
├── app.py                  # 主程序（Streamlit UI + RAG 流程 + 评测入口）
├── llm_config.py           # LLM 配置（环境变量管理）
├── 01_langchain_basics.py  # 学习：LangChain 核心概念
├── 02_document_processing.py # 学习：文档处理流水线
├── 03_embedding_rag.py     # 学习：Embedding + RAG
├── 04_structured_output.py  # 学习：结构化输出（Pydantic）
├── 05_rag_evaluation.py     # 🌟 评测脚本：量化验证 RAG 系统
├── 06_resume_jd_matcher.py  # 学习：简历 JD 匹配助手
├── docs/                   # 项目文档
│   └── 01-技术方案.md
├── outputs/                # 输出目录（向量库持久化）
│   └── vectorstore/
├── evaluation_report.md    # 评测报告（自动生成）
├── .env.example            # 环境变量模板
├── requirements.txt
└── README.md
```

## 配置说明

| 环境变量 | 必填 | 说明 | 默认值 |
|----------|------|------|--------|
| DEEPSEEK_API_KEY | ✅ | DeepSeek API Key | - |
| DEEPSEEK_API_BASE | ❌ | API 地址 | https://api.deepseek.com/v1 |
| DEEPSEEK_MODEL | ❌ | 模型名称 | deepseek-chat |
| LLM_MAX_TOKENS | ❌ | 最大输出 Token | 4096 |

## 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | - | 初始版本 |
| v2.0 | 2026-07-12 | 代码去重、错误处理、向量库持久化 |
| v2.1 | 2026-07-20 | 新增 **RAG 评测功能**，内置量化评估体系 |
