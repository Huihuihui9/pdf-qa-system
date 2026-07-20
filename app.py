"""PDF 知识库问答系统 v2.1

支持上传PDF → 自动切块向量化 → 用户提问 → 检索+LLM回答 → 展示引用来源
新增：RAG评测入口
优化：去重逻辑、错误处理、向量库持久化、环境检测
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
from dotenv import load_dotenv

# ---------- 配置 ----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
VECTORSTORE_DIR = "outputs/vectorstore"

@st.cache_resource
def get_embeddings():
    """加载Embedding模型（缓存，只加载一次）"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def process_pdf(uploaded_file):
    """处理单个PDF：加载 → 切块（返回文档块列表）"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    except Exception as e:
        st.error(f"❌ 加载 PDF 失败（{uploaded_file.name}）: {e}")
        return [], 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        page = chunk.metadata.get("page", "未知")
        chunk.metadata["source"] = f"{uploaded_file.name} - 第{page + 1}页"
        chunk.metadata["filename"] = uploaded_file.name

    return chunks, len(docs)


def save_vectorstore(vectorstore, path=VECTORSTORE_DIR):
    """持久化向量库到磁盘"""
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vectorstore(path=VECTORSTORE_DIR):
    """从磁盘加载向量库"""
    if os.path.exists(os.path.join(path, "index.faiss")):
        embeddings = get_embeddings()
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None


def check_environment():
    """启动时检查环境配置"""
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.warning(
            "⚠️ 未检测到 DEEPSEEK_API_KEY\n"
            "请在项目根目录创建 .env 文件，内容参考 .env.example",
            icon="⚠️",
        )
        return False
    return True


def build_index(uploaded_files):
    """上传PDF → 切块 → 向量化 → 建索引"""
    all_chunks = []
    total_pages = 0

    embeddings = get_embeddings()
    progress_bar = st.progress(0, text="准备处理...")

    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i) / len(uploaded_files), text=f"处理: {file.name}")
        chunks, pages = process_pdf(file)
        all_chunks.extend(chunks)
        total_pages += pages

    if not all_chunks:
        st.error("❌ 没有成功处理任何文件")
        return None, 0, 0

    progress_bar.progress(0.9, text="正在向量化并建立索引...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    progress_bar.empty()

    return vectorstore, len(all_chunks), total_pages


def ask_question(vectorstore, question, k=3):
    """检索相关文档 + 生成回答"""
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception as e:
        return f"❌ 检索失败: {e}", [], []

    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "未知") for d in docs]

    from llm_config import get_llm

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个智能文档助手。请根据以下参考资料回答用户问题。\n"
         "如果资料中没有相关信息，请如实告知。回答时请引用来源。\n\n"
         "参考资料：\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        answer = f"❌ LLM 调用失败: {e}"

    return answer, sources, docs


# ---------- 主页面 ----------
def main():
    st.set_page_config(page_title="PDF 知识库问答", page_icon="📄", layout="wide")
    st.title("📄 PDF 知识库问答系统")
    st.caption("上传PDF文件，基于文档内容进行智能问答 | v2.1")

    # 环境检查
    env_ok = check_environment()

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        saved = load_vectorstore()
        if saved is not None:
            st.session_state.vectorstore = saved
    if "run_eval" not in st.session_state:
        st.session_state["run_eval"] = False

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("📁 知识库管理")

        uploaded = st.file_uploader(
            "上传PDF文件",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持批量上传多个 PDF",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            build_btn = st.button("🚀 开始索引", use_container_width=True, disabled=not uploaded)
        with col2:
            clear_btn = st.button("🗑️ 清空索引", use_container_width=True)

        if clear_btn:
            st.session_state.vectorstore = None
            st.session_state.messages = []
            import shutil

            if os.path.exists(VECTORSTORE_DIR):
                shutil.rmtree(VECTORSTORE_DIR)
            st.rerun()

        if build_btn and uploaded:
            valid_files = [f for f in uploaded if f.name.lower().endswith(".pdf")]
            if not valid_files:
                st.error("❌ 请上传 PDF 格式的文件")
            else:
                with st.spinner("正在处理 PDF..."):
                    vectorstore, chunk_count, page_count = build_index(valid_files)

                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.messages = []
                    save_vectorstore(vectorstore)
                    st.success(
                        f"✅ 索引完成！\n"
                        f"- 文件: {len(valid_files)} 个\n"
                        f"- 总页数: {page_count}\n"
                        f"- 分块: {chunk_count}"
                    )
                    st.rerun()

        st.divider()

        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            st.info(f"📚 知识库已就绪（{st.session_state.vectorstore.index.ntotal} 个向量）")
        else:
            st.info("📭 知识库为空，请上传PDF")

        st.divider()
        if st.button("📊 运行RAG评测", use_container_width=True):
            st.session_state["run_eval"] = True

    # --- RAG 测评模式 ---
    if st.session_state["run_eval"]:
        st.subheader("📊 RAG 评测结果")
        st.info("评测功能已就绪！请在终端运行 `python 05_rag_evaluation.py` 进行量化评测，结果将自动保存至 evaluation_report.md")
        st.success("✅ 测评完成后，报告将显示 Faithfulness (忠实度) 和 Context Precision (检索精度) 指标")
        st.session_state["run_eval"] = False
        return

    # --- 主区域：对话 ---
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.info("👈 请先在左侧上传PDF文件并建立索引")
        return

    if not env_ok:
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 引用来源"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

    if question := st.chat_input("请输入关于PDF内容的问题"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索并生成回答..."):
                answer, sources, _ = ask_question(st.session_state.vectorstore, question)
            st.write(answer)
            if sources:
                with st.expander("📎 引用来源"):
                    for s in sources:
                        st.write(f"- {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
