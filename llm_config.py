import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 显式设置环境变量
load_dotenv()

# 确保使用自定义的 key 和 base_url
api_key = os.getenv("DEEPSEEK_API_KEY", "")
base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.xiaomimimo.com/v1")
model = os.getenv("DEEPSEEK_MODEL", "mimo-v2.5-pro")


def get_llm(temperature=0.7):
    """获取LLM实例"""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=1000,
    )