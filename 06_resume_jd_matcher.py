"""
补6: 简历 JD 匹配助手 — 输入简历和JD → 输出技能缺口
用途：面试前快速对比自己的技能和岗位要求
技术：Pydantic 结构化输出（复用 04 的经验）+ DeepSeek

对比普通的"人工对比"：
  AI 能更全面地找出隐性要求（如"团队协作能力""沟通能力"）
  并且自动排优先级
"""
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8')

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()


# ====== 数据结构 ======

class ResumeInfo(BaseModel):
    """简历信息"""
    name: str = Field(description="姓名")
    current_role: str = Field(description="当前职位/角色")
    years_exp: float = Field(description="工作年限")
    skills: list[str] = Field(description="掌握的技能列表")
    experience_summary: str = Field(description="工作经历简述")
    education: str = Field(description="学历")


class JDRequirement(BaseModel):
    """JD 要求"""
    position_name: str = Field(description="职位名称")
    required_skills: list[str] = Field(description="硬性必备技能")
    nice_to_have: list[str] = Field(description="加分技能")
    experience_required: str = Field(description="经验要求")
    responsibilities: list[str] = Field(description="主要职责")


class SkillGap(BaseModel):
    """单个技能缺口"""
    skill: str = Field(description="技能名称")
    status: str = Field(description="状态: '已掌握' / '部分掌握' / '未掌握'")
    priority: int = Field(description="优先级 1-5（1最高）")
    reason: str = Field(description="为什么这个技能重要")
    action: str = Field(description="建议行动")


class MatchAnalysis(BaseModel):
    """完整匹配分析"""
    total_gaps: int = Field(description="技能缺口总数")
    gaps: list[SkillGap] = Field(description="技能缺口列表")
    match_score: int = Field(description="整体匹配度 0-100")
    strengths: list[str] = Field(description="你的优势")
    overall_advice: str = Field(description="综合建议")


# ====== 工具函数 ======

def parse_json(text: str, model_class):
    """解析 LLM 输出的 JSON 到 Pydantic 模型"""
    # 清理 markdown 格式
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    # 清理可能的 trailing comma
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    try:
        data = json.loads(text.strip())
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  ❌ 解析失败: {e}")
        return None


def call_llm_json(system_prompt: str, user_input: str,
                  model_class, temperature=0.3):
    """调用 LLM 并解析为结构化数据"""
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_input),
    ])

    result = (prompt | llm | StrOutputParser()).invoke({})
    return parse_json(result, model_class)


# ====== 主流程 ======

def analyze_resume(resume_text: str) -> ResumeInfo | None:
    """提取简历中的结构化信息"""
    print("  [1/3] 分析简历...")
    system_prompt = """你是一个简历解析专家。从以下简历文本中提取结构化信息。
输出 JSON 格式，字段：
- name: 姓名
- current_role: 当前职位
- years_exp: 工作年限（数字）
- skills: 技能列表
- experience_summary: 工作经历简述
- education: 学历

只输出 JSON，不要多余文字。"""
    return call_llm_json(system_prompt, resume_text, ResumeInfo)


def analyze_jd(jd_text: str) -> JDRequirement | None:
    """提取 JD 中的结构化信息"""
    print("  [2/3] 分析 JD...")
    system_prompt = """你是一个职位分析专家。从以下 JD 中提取结构化信息。
输出 JSON 格式，字段：
- position_name: 职位名称
- required_skills: 硬性技能要求列表
- nice_to_have: 加分技能列表
- experience_required: 经验要求
- responsibilities: 主要职责列表

只输出 JSON，不要多余文字。"""
    return call_llm_json(system_prompt, jd_text, JDRequirement)


def match_resume_jd(resume: ResumeInfo, jd: JDRequirement) -> MatchAnalysis | None:
    """对比简历和 JD，生成技能缺口分析"""
    print("  [3/3] 匹配分析...")
    system_prompt = """你是一个职业发展顾问。对比以下简历和 JD 要求，
找出技能缺口并排优先级。

输出 JSON 格式：
{{
  "total_gaps": 缺口总数,
  "gaps": [
    {{"skill": "技能名", "status": "已掌握/部分掌握/未掌握",
     "priority": 优先级1-5, "reason": "为什么重要", "action": "建议行动"}}
  ],
  "match_score": 整体匹配度0-100,
  "strengths": ["优势1", "优势2"],
  "overall_advice": "综合建议"
}}

priority 1-5: 1是最高优先级（必须马上补），5是最低（锦上添花）
只输出 JSON。"""
    user_input = f"""
【简历】
姓名: {resume.name}
角色: {resume.current_role}
年限: {resume.years_exp}年
技能: {', '.join(resume.skills)}
经历: {resume.experience_summary}
学历: {resume.education}

【JD要求】
职位: {jd.position_name}
必备技能: {', '.join(jd.required_skills)}
加分技能: {', '.join(jd.nice_to_have)}
经验要求: {jd.experience_required}
职责: {', '.join(jd.responsibilities)}
"""
    return call_llm_json(system_prompt, user_input, MatchAnalysis, temperature=0.3)


def print_report(result: MatchAnalysis):
    """打印分析报告"""
    print(f"\n{'=' * 55}")
    print(f"📊 匹配分析报告")
    print(f"{'=' * 55}")

    # 匹配度
    score = result.match_score
    bar = "█" * (score // 10) + "░" * (10 - score // 10)
    print(f"\n整体匹配度: {score}%  {bar}")

    # 优势
    print(f"\n✅ 你的优势:")
    for s in result.strengths:
        print(f"  • {s}")

    # 技能缺口
    print(f"\n⚠️  技能缺口（共 {result.total_gaps} 项）:")
    print(f"{'优先级':<6} {'技能':<16} {'状态':<10} {'建议'}")
    print(f"{'-' * 55}")
    for g in sorted(result.gaps, key=lambda x: x.priority):
        priority_str = "🔴" if g.priority <= 2 else "🟡" if g.priority <= 3 else "🟢"
        print(f"{priority_str} P{g.priority:<3} {g.skill:<16} {g.status:<10} {g.action}")

    # 综合建议
    print(f"\n💡 综合建议:")
    print(f"  {result.overall_advice}")


def run_analysis(resume_text: str, jd_text: str):
    """完整流程：简历分析 → JD分析 → 匹配 → 报告"""
    print("\n" + "=" * 55)
    print("简历 JD 匹配助手")
    print("=" * 55)

    # Step 1: 提取简历信息
    resume = analyze_resume(resume_text)
    if not resume:
        print("❌ 简历分析失败")
        return

    print(f"  ✓ 识别到: {resume.name} | {resume.current_role} | {resume.years_exp}年经验")

    # Step 2: 提取 JD 信息
    jd = analyze_jd(jd_text)
    if not jd:
        print("❌ JD 分析失败")
        return

    print(f"  ✓ 识别到: {jd.position_name} | {len(jd.required_skills)}项必备技能")

    # Step 3: 匹配分析
    result = match_resume_jd(resume, jd)
    if not result:
        print("❌ 匹配分析失败")
        return

    print_report(result)


# ====== 预设案例 ======

SAMPLE_RESUME = """
姓名：张三
当前职位：初级 Python 开发工程师
工作年限：1.5年
技能：Python基础、FastAPI基础、Git、Linux基础、MySQL基础
经历：在一家创业公司做后端开发，负责 RESTful API 开发和数据库维护
学历：本科（计算机科学与技术）
"""

SAMPLE_JD = """
职位：AI应用开发工程师
薪资：25K-40K·15薪
要求：
- 2年以上Python开发经验
- 熟悉 LangChain 或 LlamaIndex
- 有 RAG 和向量数据库开发经验
- 熟悉 FastAPI
- 了解 Agent/Tool Calling 机制
加分：
- 有 Docker 使用经验
- 了解 MCP 协议
- 有 Streamlit 开发经验
- 了解 SQL 和数据库设计
"""


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("简历 JD 匹配助手")
    print("=" * 55)
    print()
    print("功能：输入简历 → 输入 JD → AI 自动分析技能缺口")
    print()

    # 使用预设案例
    use_sample = input("使用预设案例？(y/n, 默认 y): ").strip().lower()
    if use_sample != "n":
        resume_text = SAMPLE_RESUME
        jd_text = SAMPLE_JD
        print(f"\n📄 简历: {SAMPLE_RESUME.strip()}")
        print(f"📋 JD: {SAMPLE_JD.strip()}")
    else:
        print("\n请输入简历文本（多行，输入 END 结束）:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        resume_text = "\n".join(lines)

        print("\n请输入 JD 文本（多行，输入 END 结束）:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        jd_text = "\n".join(lines)

    run_analysis(resume_text, jd_text)
