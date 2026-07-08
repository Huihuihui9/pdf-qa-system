"""
LangChain 核心概念学习
概念1：Prompt Template — 提示词模板
概念2：Chain — 把多个步骤串起来
概念3：Memory — 让对话有上下文
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from llm_config import get_llm


def demo_prompt_template():
    """概念1：Prompt Template"""
    print("=== Prompt Template ===\n")
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用{style}的语气回答。控制在100字以内。"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    roles = [
        {"role": "厨师", "style": "热情", "question": "怎么做番茄炒蛋？"},
        {"role": "健身教练", "style": "严格", "question": "怎么减肚子上的脂肪？"},
    ]

    for params in roles:
        result = chain.invoke(params)
        print(f"角色：{params['role']}")
        print(f"回答：{result}\n")


def demo_chain():
    """概念2：Chain — 多步串联"""
    print("=== Chain（多步处理）===\n")
    llm = get_llm(temperature=0.3)

    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "把用户输入的中文翻译成英文，只输出翻译结果"),
        ("human", "{text}"),
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in one sentence. Output in Chinese."),
        ("human", "{text}"),
    ])

    translate_chain = translate_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()

    text = "LangChain是一个开源框架，帮助开发者将大语言模型与外部数据和工具集成。"

    english = translate_chain.invoke({"text": text})
    print(f"原文：{text}")
    print(f"翻译：{english}")

    summary = summary_chain.invoke({"text": english})
    print(f"摘要：{summary}\n")


def demo_memory():
    """概念3：Memory — 对话记忆"""
    print("=== Memory（多轮对话）===\n")
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。请简洁回答。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    history = []

    questions = [
        "我叫小明，我是一个Python初学者",
        "你能推荐一些适合我的学习资源吗？",
        "我刚才说我叫什么？我是什么水平？",
    ]

    for q in questions:
        print(f"用户：{q}")
        result = chain.invoke({"history": history, "question": q})
        print(f"AI：{result}\n")
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=result))


if __name__ == "__main__":
    print("LangChain 核心概念学习\n")
    demo_prompt_template()
    print("-" * 50)
    demo_chain()
    print("-" * 50)
    demo_memory()