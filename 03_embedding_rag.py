"""
Day 19-21: 向量化 + 检索 + RAG完整链路
概念：Embedding = 把文字变成数字，语义相似的文字向量距离近
流程：文档 → 切块 → 向量化 → 存入向量库 → 检索 → 生成回答
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm_config import get_llm

# 使用本地中文Embedding模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def create_vectorstore():
    """创建向量数据库示例"""
    print("=== 创建向量数据库 ===\n")

    # 模拟知识库文档
    documents = [
        "Python是一种解释型、面向对象的高级编程语言，由Guido van Rossum于1991年发布。Python的设计哲学强调代码的可读性和简洁的语法。",
        "LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，包括Prompt管理、Chain、Agent和Memory等。",
        "RAG（检索增强生成）是一种将信息检索与文本生成相结合的技术。它先从知识库中检索相关文档，再将文档和问题一起发给LLM生成回答。",
        "FAISS是Facebook AI开发的高效向量相似度搜索库，支持十亿级别的向量搜索，常用于RAG系统的向量存储和检索。",
        "Embedding模型将文本转换为固定维度的向量表示，语义相似的文本在向量空间中距离更近。常用的中文Embedding模型包括BGE和M3E。",
        "Streamlit是一个开源的Python库，可以快速将数据脚本转换为可分享的Web应用，常用于构建AI项目的演示界面。",
        "Docker是一个容器化平台，可以将应用及其依赖打包成一个独立的容器，确保在任何环境中都能一致运行。",
    ]

    # 1. 文本切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50
    )
    texts = splitter.create_documents(documents)
    print(f"文档切块数: {len(texts)}")

    # 2. 加载Embedding模型
    print(f"加载Embedding模型: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. 向量化并存入FAISS
    print("创建向量索引...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print(f"向量库创建完成，共 {vectorstore.index.ntotal} 个向量\n")

    return vectorstore


def demo_similarity_search(vectorstore):
    """演示相似度检索"""
    print("=== 相似度检索 ===\n")

    queries = [
        "Python是什么时候发布的？",
        "怎么把大模型和外部知识结合起来？",
        "有什么工具可以快速做AI网页界面？",
    ]

    for q in queries:
        docs = vectorstore.similarity_search(q, k=2)
        print(f"问题: {q}")
        for i, doc in enumerate(docs):
            print(f"  检索结果{i+1}: {doc.page_content[:80]}...")
        print()


def demo_rag(vectorstore):
    """演示完整RAG链路"""
    print("=== 完整RAG链路 ===\n")

    llm = get_llm(temperature=0.3)

    questions = [
        "什么是LangChain？它有哪些核心组件？",
        "RAG是什么？它的主要流程是怎样的？",
    ]

    for question in questions:
        print(f"用户问题: {question}\n")

        # 1. 检索相关文档
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join(d.page_content for d in docs)
        print(f"检索到 {len(docs)} 个相关文档块")

        # 2. 构建增强Prompt
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请如实告知。\n\n参考资料：\n{context}"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        # 3. 生成回答
        answer = chain.invoke({"context": context, "question": question})
        print(f"RAG回答: {answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    vectorstore = create_vectorstore()
    demo_similarity_search(vectorstore)
    demo_rag(vectorstore)