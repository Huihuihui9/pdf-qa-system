import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature=0.7, max_tokens=None):
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置！")

    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
