# PDF 知识库问答系统

基于 LangChain + RAG 的本地PDF知识库问答系统。

## 功能

- 支持上传多个PDF文件
- 自动切块并建立向量索引（FAISS）
- 用户输入问题，系统检索相关段落并回答
- 支持查看引用来源（哪个PDF的哪一页）
- 多轮对话，保持上下文

## 技术栈

- LangChain：LLM编排框架
- FAISS：向量数据库
- BGE-small-zh：中文Embedding模型（本地运行）
- Streamlit：Web界面

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API
cp .env.example .env  # 填入你的API Key

# 启动
streamlit run app.py
```

## 项目结构

```
pdf-qa-system/
├── app.py              # Streamlit主界面
├── llm_config.py       # LLM配置
├── 01_langchain_basics.py   # 学习：LangChain核心概念
├── 02_document_processing.py # 学习：文档处理流水线
├── 03_embedding_rag.py      # 学习：Embedding+RAG
├── requirements.txt
└── README.md
```
