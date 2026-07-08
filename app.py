"""
PDF 知识库问答系统
支持上传PDF → 自动切块向量化 → 用户提问 → 检索+LLM回答 → 展示引用来源
"""
import os
import tempfile
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- 配置 ----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


@st.cache_resource
def get_embeddings():
    """加载Embedding模型（只加载一次）"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def process_pdf(uploaded_file):
    """处理上传的PDF：切块 → 向量化 → 建索引"""
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    # 加载PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # 切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 为每个chunk添加来源信息
    for chunk in chunks:
        page = chunk.metadata.get("page", "未知")
        chunk.metadata["source"] = f"第{page + 1}页"

    # 向量化
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 清理临时文件
    os.unlink(tmp_path)

    return vectorstore, len(chunks), len(docs)


def ask_question(vectorstore, question, k=3):
    """检索相关文档并回答"""
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "未知") for d in docs]

    # 获取LLM
    from llm_config import get_llm
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能文档助手。请根据以下参考资料回答用户问题。\n"
                   "如果资料中没有相关信息，请如实告知。回答时请引用来源。\n\n"
                   "参考资料：\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, sources, docs


# ---------- 页面 ----------
def main():
    st.set_page_config(page_title="PDF 知识库问答", page_icon="📄", layout="wide")
    st.title("📄 PDF 知识库问答系统")
    st.caption("上传PDF文件，基于文档内容进行智能问答")

    # 初始化session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 侧边栏：上传PDF
    with st.sidebar:
        st.header("📁 知识库管理")
        uploaded = st.file_uploader("上传PDF文件", type=["pdf"], accept_multiple_files=True)

        if uploaded and st.button("开始索引"):
            all_chunks = 0
            all_pages = 0
            embeddings = get_embeddings()
            vectorstore = None

            with st.spinner("正在处理PDF..."):
                for file in uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(file.read())
                        tmp_path = f.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    all_pages += len(docs)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=100,
                        separators=["\n\n", "\n", "。", "，", " ", ""],
                    )
                    chunks = splitter.split_documents(docs)
                    for c in chunks:
                        c.metadata["source"] = f"{file.name} - 第{c.metadata.get('page', 0) + 1}页"
                    all_chunks += len(chunks)

                    os.unlink(tmp_path)

                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                    else:
                        vectorstore.add_documents(chunks)

            st.session_state.vectorstore = vectorstore
            st.success(f"✅ 索引完成！\n- 文件: {len(uploaded)} 个\n- 页数: {all_pages}\n- 分块: {all_chunks}")

    # 主区域：对话
    if "vectorstore" not in st.session_state:
        st.info("👈 请先在左侧上传PDF文件并建立索引")
        return

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 引用来源"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

    # 用户输入
    question = st.chat_input("请输入关于PDF内容的问题")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                answer, sources, docs = ask_question(
                    st.session_state.vectorstore, question
                )
            st.write(answer)
            with st.expander("📎 引用来源"):
                for s in sources:
                    st.write(f"- {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()