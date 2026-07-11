import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature=0.7, max_tokens=None):
    \"\"\"获取LLM实例

    通过 .env 文件配置以下环境变量：
      - DEEPSEEK_API_KEY  (必填)
      - DEEPSEEK_API_BASE (可选，默认 https://api.deepseek.com/v1)
      - DEEPSEEK_MODEL    (可选，默认 deepseek-chat)
      - LLM_MAX_TOKENS    (可选，覆盖 max_tokens)

    Args:
        temperature: 生成温度 (0.0-1.0)，默认 0.7
        max_tokens:  最大输出 token 数，None 时从环境变量或默认值获取

    Returns:
        ChatOpenAI 实例

    Raises:
        ValueError: 当 DEEPSEEK_API_KEY 未设置时
    \"\"\"
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置！请在项目根目录创建 .env 文件，"
            "内容参考 .env.example"
        )

    base_url = os.getenv(
        "DEEPSEEK_API_BASE",
        "https://api.deepseek.com/v1",
    )
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if max_tokens is None:
        max_tokens_env = os.getenv("LLM_MAX_TOKENS", "4096")
        max_tokens = int(max_tokens_env)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
