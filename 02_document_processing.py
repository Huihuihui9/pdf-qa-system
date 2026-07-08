"""
Day 17-18: 文档处理流水线
Document Loader + Text Splitter
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def demo_text_splitter():
    """文本切块演示"""
    print("=== 文本切块 ===\n")

    sample = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。

近年来，深度学习技术取得了突破性进展。2017年，Transformer架构的提出彻底改变了NLP领域。
基于Transformer的大语言模型（LLM）如GPT、Claude等展现了惊人的语言理解和生成能力。

LangChain是一个开源框架，为开发者提供了构建LLM应用的工具。核心概念包括：
1. Prompt Template：模板化提示词
2. Chain：将多个处理步骤串联
3. Agent：让LLM自主决定调用哪些工具
4. Memory：在对话中保持上下文

RAG是一种将外部知识库与LLM结合的技术。基本流程：
1. 将文档切分成小块
2. 将每个小块转换为向量（Embedding）
3. 用户提问时，检索最相关的文档块
4. 将检索到的内容和问题一起发给LLM
5. LLM基于检索到的内容生成回答

这种方法的优势在于不需要微调模型，知识可以实时更新。
"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_text(sample)
    print(f"原文: {len(sample)} 字符 → {len(chunks)} 块\n")

    for i, chunk in enumerate(chunks):
        print(f"--- 第{i+1}块 ({len(chunk)}字符) ---")
        print(chunk)
        print()


def demo_chunk_params():
    """不同参数对比"""
    print("\n=== 参数对比 ===\n")
    text = "人工智能是计算机科学的一个分支。" * 20

    configs = [
        {"chunk_size": 100, "chunk_overlap": 0},
        {"chunk_size": 100, "chunk_overlap": 30},
        {"chunk_size": 200, "chunk_overlap": 50},
    ]

    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        chunks = splitter.split_text(text)
        print(f"chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']} → {len(chunks)} 块")
        print(f"  第1块: {chunks[0][:60]}...\n")


def demo_pdf_loading():
    """PDF加载演示"""
    print("\n=== PDF加载 ===\n")
    pdf_path = "example.pdf"

    import os
    if not os.path.exists(pdf_path):
        print(f"文件 {pdf_path} 不存在，跳过")
        print("使用方法: loader = PyPDFLoader('your_file.pdf')\n  docs = loader.load()")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"总页数: {len(docs)}")
    for doc in docs[:3]:
        print(f"\n--- 第{doc.metadata.get('page', '?')}页 ---")
        print(doc.page_content[:200])


if __name__ == "__main__":
    demo_text_splitter()
    demo_chunk_params()
    demo_pdf_loading()