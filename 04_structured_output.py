"""
补1: 结构化输出 — 让LLM稳定输出JSON
面试必问："你怎么保证模型输出格式是稳定的？"
核心工具：Pydantic + LangChain with_structured_output
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pydantic import BaseModel, Field

from llm_config import get_llm


def demo_pydantic_basics():
    """Pydantic基础：定义数据模型"""
    print("=== Pydantic 数据模型 ===")

    # 定义数据结构，指定字段类型和描述
    class ResumeInfo(BaseModel):
        name: str = Field(description="候选人姓名")
        age: int = Field(description="年龄")
        skills: list[str] = Field(description="技能列表")
        years_experience: float = Field(description="工作年限")

    class JobPost(BaseModel):
        title: str = Field(description="职位名称")
        salary_range: str = Field(description="薪资范围")
        required_skills: list[str] = Field(description="必备技能")
        education_required: str = Field(description="学历要求")

    # Pydantic 自动校验类型
    resume = ResumeInfo(name="张三", age=28, skills=["Python", "SQL", "LangChain"], years_experience=3.0)
    print(f"\n简历对象: {resume.model_dump()}")
    print(f"姓名: {resume.name}")
    print(f"JSON:\n{resume.model_dump_json(indent=2)}")

    # 错误类型会报错
    try:
        bad_data = ResumeInfo(name="李四", age="不是数字", skills=[], years_experience="abc")
    except Exception as e:
        print(f"\n类型校验拦截: {e}")


def demo_structured_output():
    """让LLM稳定输出JSON"""
    print("\n=== Structured Outputs ===\n")

    llm = get_llm(temperature=0.3)

    class JobAnalysis(BaseModel):
        """JD分析结果"""
        position_name: str = Field(description="职位名称")
        salary_min: int = Field(description="最低薪资(万/年)")
        salary_max: int = Field(description="最高薪资(万/年)")
        required_skills: list[str] = Field(description="必备技能清单")
        optional_skills: list[str] = Field(description="加分技能清单")
        experience_required: str = Field(description="经验要求")
        suggestion_score: int = Field(description="匹配度评分(1-10)")

    # 方法1: 在prompt里要求JSON格式
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个JD分析专家。请分析以下JD，用JSON格式返回，必须包含以下字段：\n"
                   "position_name（职位名称）, salary_min（最低薪资万/年）, salary_max（最高薪资万/年）, "
                   "required_skills（必备技能列表）, optional_skills（加分技能列表）, "
                   "experience_required（经验要求）, suggestion_score（匹配度评分1-10）\n"
                   "只输出JSON，不要多余文字。"),
        ("human", "JD: {jd_text}"),
    ])

    chain = prompt | llm | StrOutputParser()

    jd = """高级AI应用开发工程师
    薪资：30-45K·15薪
    职责：负责RAG知识库系统开发和Agent相关功能开发
    要求：
    - 3年以上Python开发经验
    - 熟悉LangChain或LlamaIndex
    - 了解RAG和向量数据库
    - 有FastAPI开发经验
    加分：
    - 有Agent开发经验
    - 了解Docker"""

    result = chain.invoke({"jd_text": jd})
    print("方法1: Prompt要求JSON格式")
    print(f"原始输出:\n{result}\n")

    # 解析并验证
    try:
        parsed = json.loads(result)
        validated = JobAnalysis(**parsed)
        print(f"✓ 校验通过: {validated.position_name}")
        print(f"  薪资: {validated.salary_min}-{validated.salary_max}万")
        print(f"  必备技能: {validated.required_skills}")
    except Exception as e:
        print(f"✗ 解析失败: {e}")

    # 方法2: LangChain的with_structured_output（推荐）
    print("\n方法2: with_structured_output (推荐)")
    structured_llm = llm.with_structured_output(JobAnalysis)

    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "分析以下JD并提取结构化信息。"),
        ("human", "JD: {jd_text}"),
    ])

    chain2 = prompt2 | structured_llm
    result2 = chain2.invoke({"jd_text": jd})

    print(f"✓ 类型: {type(result2).__name__}")
    print(f"  职位: {result2.position_name}")
    print(f"  薪资: {result2.salary_min}-{result2.salary_max}万")
    print(f"  必备技能: {result2.required_skills}")
    print(f"  匹配度: {result2.suggestion_score}/10")


def demo_multi_output():
    """批量结构化输出"""
    print("\n=== 批量结构化输出 ===\n")

    class SkillGap(BaseModel):
        skill: str = Field(description="技能名称")
        current_level: str = Field(description="当前水平")
        target_level: str = Field(description="目标水平")
        priority: int = Field(description="优先级1-5")
        learning_resource: str = Field(description="推荐学习资源")

    class SkillAnalysis(BaseModel):
        total_gaps: int = Field(description="技能缺口总数")
        gaps: list[SkillGap] = Field(description="技能缺口列表")
        overall_advice: str = Field(description="整体建议")

    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(SkillAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "分析简历技能和JD要求之间的差距。返回结构化分析结果。"),
        ("human", "简历技能: Python基础, FastAPI基础\nJD要求: Python精通, LangChain, RAG, Agent开发, Docker, SQL"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({})

    print(f"技能缺口总数: {result.total_gaps}")
    print(f"\n整体建议: {result.overall_advice}\n")
    for gap in result.gaps:
        print(f"  [{gap.priority}] {gap.skill}: {gap.current_level} → {gap.target_level}")
        print(f"      推荐: {gap.learning_resource}")


if __name__ == "__main__":
    demo_pydantic_basics()
    demo_structured_output()
    demo_multi_output()
