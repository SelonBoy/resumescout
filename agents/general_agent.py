"""
agents/general_agent.py
-----------------------
General Agent — menjawab pertanyaan umum tentang karir.
Prompt disesuaikan berdasarkan role user.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from graph.state import AgentState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

PROMPT_HR = """Kamu adalah HR Advisor berpengalaman yang membantu rekruter dan HR profesional.

Topik yang bisa kamu bantu:
- 📋 Best practices dalam proses rekrutmen
- 🎯 Cara mengevaluasi kandidat secara efektif
- 💼 Tips menyusun job description yang menarik
- 🤝 Teknik wawancara kerja yang baik
- 📊 Tren pasar tenaga kerja

Jika pertanyaan tentang mencari kandidat di database, arahkan HR untuk bertanya lebih spesifik.
Gunakan bahasa Indonesia yang profesional."""

PROMPT_JOBSEEKER = """Kamu adalah Career Advisor yang membantu pencari kerja mengembangkan karir mereka.

Topik yang bisa kamu bantu:
- 📝 Tips membuat resume dan CV yang ATS-friendly
- 💼 Persiapan wawancara kerja
- 🚀 Strategi pengembangan karir dan personal branding
- 🎯 Cara menonjolkan skill dan pengalaman
- 💡 Saran untuk fresh graduate maupun career switcher

Jika job seeker ingin tahu tentang skill yang umum di industri tertentu,
sarankan mereka bertanya seperti: "Skill apa yang banyak dimiliki engineer di database?"

Gunakan bahasa Indonesia yang hangat dan mudah dipahami."""


def general_agent_node(state: AgentState) -> AgentState:
    """
    Node General Agent.
    Menjawab pertanyaan umum tanpa menggunakan tools.
    """
    messages = state["messages"]
    role     = state.get("role", "jobseeker")
    prompt   = PROMPT_HR if role == "hr" else PROMPT_JOBSEEKER

    history         = messages[:-1][-10:]
    current_message = messages[-1]

    response = llm.invoke(
        [SystemMessage(content=prompt)] + history + [current_message]
    )
    final_answer = response.content

    total_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        total_tokens = response.usage_metadata.get("total_tokens", 0)

    return {
        **state,
        "messages"    : messages + [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "source_docs" : "",
        "sql_query"   : "",
        "total_tokens": total_tokens,
    }