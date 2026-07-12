"""
补1: 结构化输出（DeepSeek兼容版）
核心思路：Prompt要求输出JSON → Pydantic自动校验
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm


def demo_pydantic_basics():
    """Pydantic基础"""
    print("=== Pydantic: 定义数据结构 + 自动校验 ===")

    class ResumeInfo(BaseModel):
        name: str = Field(description="候选人姓名")
        age: int = Field(description="年龄")
        skills: list[str] = Field(description="技能列表")

    r = ResumeInfo(name="张三", age=28, skills=["Python", "LangChain"])
    print(f"✓ 校验通过: {r.model_dump()}")
    print(f"  JSON: {r.model_dump_json(indent=2)}")

    # 类型错误会被拦截
    try:
        ResumeInfo(name="李四", age="不是数字", skills=[])
    except ValidationError as e:
        print(f"✓ 类型错误拦截: age 字段类型不对\n")


def parse_json_to_model(text: str, model_class):
    """把JSON字符串解析成Pydantic模型"""
    try:
        # 清理可能的 markdown 代码块标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  解析失败: {e}")
        return None


def demo_structured_output():
    """Prompt要求JSON格式 + Pydantic校验"""
    print("=== 结构化输出: Prompt → JSON → Pydantic ===\n")

    # 1. 定义数据结构
    class JobAnalysis(BaseModel):
        position_name: str = Field(description="职位名称")
        salary_min: int = Field(description="最低薪资(万/年)")
        salary_max: int = Field(description="最高薪资(万/年)")
        required_skills: list[str] = Field(description="必备技能")
        experience_required: str = Field(description="经验要求")
        suggestion_score: int = Field(description="匹配度(1-10)")

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个JD分析专家。分析以下JD并严格输出JSON，格式：\n"
                   '{{"position_name":"职位名","salary_min":最低薪资,"salary_max":最高薪资,'
                   '"required_skills":["技能1","技能2"],"experience_required":"经验要求",'
                   '"suggestion_score":匹配度}}\n'
                   "只输出JSON，不要多余文字。"),
        ("human", "JD: {jd_text}"),
    ])

    chain = prompt | llm | StrOutputParser()

    jd = """高级AI应用开发工程师 | 薪资30-45K·15薪 | 要求：3年+Python开发，
    熟悉LangChain、RAG、向量数据库、FastAPI。加分：Agent开发、Docker。"""

    raw_result = chain.invoke({"jd_text": jd})
    print(f"LLM原始输出:\n{raw_result}\n")

    # 用Pydantic校验
    parsed = parse_json_to_model(raw_result, JobAnalysis)
    if parsed:
        print(f"✓ 校验通过: {parsed.position_name}")
        print(f"  薪资: {parsed.salary_min}-{parsed.salary_max}万")
        print(f"  必备技能: {parsed.required_skills}")
        print(f"  匹配度: {parsed.suggestion_score}/10")


def demo_multi_output():
    """批量结构化输出（技能缺口分析）"""
    print("\n=== 批量结构化: 简历vsJD分析 ===\n")

    class SkillGap(BaseModel):
        skill: str = Field(description="技能名称")
        priority: int = Field(description="优先级1-5")
        reason: str = Field(description="为什么需要学这个")

    class Analysis(BaseModel):
        gaps: list[SkillGap] = Field(description="技能缺口列表")
        total_gaps: int = Field(description="缺口总数")
        overall_advice: str = Field(description="整体建议")

    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "分析简历和JD要求之间的缺口，输出JSON：\n"
                   '{{"gaps":[{{"skill":"技能名","priority":优先级,"reason":"原因"}}],'
                   '"total_gaps":缺口数,"overall_advice":"整体建议"}}\n'
                   "只输出JSON。"),
        ("human", "简历: Python基础, FastAPI基础, 会Git\n"
                  "JD要求: Python精通, LangChain, RAG, Agent, Docker, SQL"),
    ])

    raw = (prompt | llm | StrOutputParser()).invoke({})
    print(f"LLM原始输出:\n{raw}\n")

    parsed = parse_json_to_model(raw, Analysis)
    if parsed:
        print(f"✓ 共{parsed.total_gaps}个技能缺口\n")
        print("整体建议:", parsed.overall_advice)
        for g in parsed.gaps:
            print(f"  优先级{g.priority}: {g.skill} — {g.reason}")


if __name__ == "__main__":
    demo_pydantic_basics()
    demo_structured_output()
    demo_multi_output()
