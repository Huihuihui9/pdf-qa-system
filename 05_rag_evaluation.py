"""
补3: RAG 评测 — 用 Ragas 量化评估 RAG 系统
指标：忠实度 (faithfulness)、检索精度 (context_precision)

原理：准备测试集 → 让RAG系统回答问题 → 用Ragas算分 → 找优化方向

注意：DeepSeek API 不支持 n>1，所以 Ragas 的部分指标不可用。
我们做两个层次评估：
  1. Ragas 标准指标（faithfulness / context_precision — 需DeepSeek兼容）
  2. 自建评估：精确命中率、检索相关率（不依赖LLM，稳定运行）
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# 临时修复：Ragas 0.4.x 会尝试导入 langchain_community.chat_models.vertexai
from unittest.mock import MagicMock
sys.modules.setdefault('langchain_community.chat_models.vertexai', MagicMock())

from datasets import Dataset
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ====== 知识库文档（和 03_embedding_rag.py 保持一致） ======
DOCUMENTS = [
    "Python是一种解释型、面向对象的高级编程语言，由Guido van Rossum于1991年发布。Python的设计哲学强调代码的可读性和简洁的语法。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，包括Prompt管理、Chain、Agent和Memory等。",
    "RAG（检索增强生成）是一种将信息检索与文本生成相结合的技术。它先从知识库中检索相关文档，再将文档和问题一起发给LLM生成回答。",
    "FAISS是Facebook AI开发的高效向量相似度搜索库，支持十亿级别的向量搜索，常用于RAG系统的向量存储和检索。",
    "Embedding模型将文本转换为固定维度的向量表示，语义相似的文本在向量空间中距离更近。常用的中文Embedding模型包括BGE和M3E。",
    "Streamlit是一个开源的Python库，可以快速将数据脚本转换为可分享的Web应用，常用于构建AI项目的演示界面。",
    "Docker是一个容器化平台，可以将应用及其依赖打包成一个独立的容器，确保在任何环境中都能一致运行。",
]

# ====== 测试集：问题 + 标准答案 ======
TEST_SET = [
    {
        "question": "Python是谁开发的？什么时候发布的？",
        "ground_truth": "Python由Guido van Rossum开发，于1991年发布。",
    },
    {
        "question": "LangChain提供了哪些核心组件？",
        "ground_truth": "LangChain提供了Prompt管理、Chain、Agent和Memory等模块化组件。",
    },
    {
        "question": "RAG的全称是什么？它的基本流程是怎样的？",
        "ground_truth": "RAG是检索增强生成（Retrieval Augmented Generation），先检索相关文档，再将文档和问题一起发给LLM生成回答。",
    },
    {
        "question": "FAISS是什么？它有什么特点？",
        "ground_truth": "FAISS是Facebook AI开发的高效向量相似度搜索库，支持十亿级别的向量搜索，常用于RAG系统。",
    },
    {
        "question": "Embedding模型的作用是什么？",
        "ground_truth": "Embedding模型将文本转换为固定维度的向量表示，语义相似的文本在向量空间中距离更近。",
    },
]

# ====== 评估用的关键词（用于自建评估） ======
# 每个问题的答案中必须包含的关键词
ANSWER_KEYWORDS = {
    0: ["Guido van Rossum", "1991"],
    1: ["Prompt管理", "Chain", "Agent", "Memory"],
    2: ["检索增强生成", "检索", "生成"],
    3: ["Facebook AI", "向量相似度", "十亿"],
    4: ["向量表示", "语义相似"],
}


def get_eval_llm(temperature=0):
    """创建用于评测的 LLM（显式传入 DeepSeek 配置）"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        temperature=temperature,
    )


def build_vectorstore():
    """构建和 03_embedding_rag.py 相同的向量库"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.create_documents(DOCUMENTS)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(texts, embeddings), embeddings


def run_rag_pipeline(vectorstore, question: str, k=3):
    """执行完整的 RAG 流程，返回 (answer, contexts)"""
    llm = get_eval_llm(temperature=0.3)

    # 1. 检索
    docs = vectorstore.similarity_search(question, k=k)
    contexts = [d.page_content for d in docs]
    context_str = "\n\n".join(contexts)

    # 2. 生成回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手。请根据以下参考资料回答问题。"
                   "如果资料中没有相关信息，请如实告知。\n\n参考资料：\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    return answer, contexts


def evaluate_builtin(answers, contexts_list, ground_truths):
    """自建评估：不依赖LLM，只做关键词匹配和检索相关度分析"""
    print("\n   自建评估（不依赖LLM，稳定可靠）")

    total = len(answers)
    keyword_hits = 0
    recall_at_1 = 0  # 第一个结果是否相关
    recall_at_3 = 0  # 前三个结果中是否有相关

    for i in range(total):
        answer = answers[i]
        contexts = contexts_list[i]
        keywords = ANSWER_KEYWORDS[i]

        # 1) 关键词命中率
        all_found = all(kw in answer for kw in keywords)
        if all_found:
            keyword_hits += 1

        # 2) 检索相关率：第一个文档是否包含关键信息
        first_doc = contexts[0] if contexts else ""
        first_relevant = any(kw in first_doc for kw in keywords)
        if first_relevant:
            recall_at_1 += 1

        # 3) 前3个文档中是否有相关
        top3_relevant = any(
            any(kw in doc for kw in keywords)
            for doc in contexts[:3]
        )
        if top3_relevant:
            recall_at_3 += 1

    return {
        "keyword_accuracy": keyword_hits / total,
        "recall_at_1": recall_at_1 / total,
        "recall_at_3": recall_at_3 / total,
    }


def try_ragas_evaluation(dataset, embeddings_model):
    """尝试用 Ragas 评估（DeepSeek 可能不支持部分指标）"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision, answer_relevancy

    eval_llm = get_eval_llm(temperature=0)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision, answer_relevancy],
            llm=eval_llm,
            embeddings=embeddings_model,
        )
        return {
            "faithfulness": result["faithfulness"],
            "context_precision": result["context_precision"],
            "answer_relevancy": result["answer_relevancy"],
        }
    except Exception as e:
        print(f"  ⚠️ Ragas 使用 DeepSeek 评测失败（DeepSeek 不支持 n>1 参数）")
        print(f"  错误: {e}")
        return None


def evaluate_rag():
    """主评测流程"""
    print("=" * 60)
    print("RAG 系统评测")
    print("=" * 60)

    # Step 1: 建向量库
    print("\n[1/4] 构建向量库...")
    vectorstore, embeddings_model = build_vectorstore()
    print(f"  ✓ 向量库创建完成，共 {vectorstore.index.ntotal} 个向量")

    # Step 2: 运行 RAG 流程
    print("\n[2/4] 运行 RAG 流程获取回答和上下文...")
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(TEST_SET):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(TEST_SET)}] {q}")

        answer, contexts = run_rag_pipeline(vectorstore, q)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)

        print(f"      回答: {answer[:80]}...")

    # Step 3: 自建评估（不依赖LLM）
    print("\n[3/4] 自建评估（关键词匹配 + 检索相关度）...")
    builtin_results = evaluate_builtin(answers, contexts_list, ground_truths)

    # Step 4: Ragas 评估（可选）
    print("\n[4/4] Ragas 评测...")
    print("  (使用 DeepSeek 作为评测 LLM)\n")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    ragas_results = try_ragas_evaluation(dataset, embeddings_model)

    # ====== 输出报告 ======
    print("\n" + "=" * 60)
    print("📊 评测报告")
    print("=" * 60)

    # 逐条显示
    print(f"\n{'─' * 40}")
    print("逐条结果：")
    print(f"{'─' * 40}")
    for i in range(len(answers)):
        print(f"\n  Q{i+1}: {questions[i]}")
        print(f"  回答: {answers[i][:100]}...")
        print(f"  命中关键词: {ANSWER_KEYWORDS[i]}")

    # 自建评估汇总
    print(f"\n{'─' * 40}")
    print("📈 自建评估汇总")
    print(f"{'─' * 40}")
    print(f"  关键词准确率 (Keyword Accuracy): {builtin_results['keyword_accuracy']:.0%}")
    print(f"  检索命中率@1 (Recall@1):        {builtin_results['recall_at_1']:.0%}")
    print(f"  检索命中率@3 (Recall@3):        {builtin_results['recall_at_3']:.0%}")
    print(f"{'─' * 40}")

    if ragas_results:
        print(f"\n📈 Ragas 指标（如有 N/A 说明 DeepSeek 不支持）")
        print(f"{'─' * 40}")
        for k, v in ragas_results.items():
            label = {"faithfulness": "忠实度", "context_precision": "检索精度",
                     "answer_relevancy": "答案相关性"}.get(k, k)
            print(f"  {label} ({k}): {v}")
        print(f"{'─' * 40}")

    # 指标解读
    print("\n🔍 解读：")
    print(f"  - Keyword Accuracy  回答的关键词覆盖度（越高越好）")
    print(f"  - Recall@1          第一个检索结果是否相关")
    print(f"  - Recall@3          前3个结果中是否有相关文档")
    print(f"  - Faithfulness      回答是否基于检索内容（需DeepSeek支持）")
    print(f"  - Context Precision 检索结果排序质量（需DeepSeek支持）")

    return builtin_results


if __name__ == "__main__":
    evaluate_rag()
