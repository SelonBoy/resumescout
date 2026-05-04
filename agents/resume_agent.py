"""
agents/resume_agent.py
----------------------
Resume RAG Agent — mencari resume dari Qdrant menggunakan ReAct agent.
Prompt disesuaikan berdasarkan role user:
  - HR        : fokus mencari kandidat terbaik
  - Job Seeker: fokus memberikan insight dan benchmark skill
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from rag.tools import ALL_TOOLS
from graph.state import AgentState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Prompt untuk HR — fokus rekrutmen
PROMPT_HR = """Kamu adalah Resume Search Specialist yang membantu HR dan rekruter menemukan kandidat terbaik.

Tools yang tersedia:
- **semantic_search_tool**: pencarian bebas berdasarkan deskripsi kualifikasi
- **category_search_tool**: pencarian dengan filter kategori pekerjaan di Qdrant

Strategi:
1. Jika HR menyebut kategori spesifik → gunakan category_search_tool
2. Jika hanya deskripsi umum → gunakan semantic_search_tool
3. Bisa kombinasikan keduanya untuk hasil lebih komprehensif

Setelah mendapat hasil:
✅ Rangkum kandidat dengan jelas dan terstruktur
✅ Highlight skill dan pengalaman yang relevan
✅ Sebutkan jumlah kandidat yang ditemukan
✅ Berikan rekomendasi kandidat terbaik berdasarkan relevansi score

Gunakan bahasa Indonesia yang profesional."""

# Prompt untuk Job Seeker — fokus insight & benchmark
PROMPT_JOBSEEKER = """Kamu adalah Career Insight Specialist yang membantu job seeker memahami lanskap karir.

Tools yang tersedia:
- **semantic_search_tool**: mencari contoh resume untuk melihat skill dan pengalaman umum
- **category_search_tool**: melihat contoh resume di bidang tertentu

Gunakan tools untuk memberikan INSIGHT kepada job seeker, seperti:
✅ Skill apa yang paling umum dimiliki kandidat di bidang X
✅ Pengalaman seperti apa yang banyak ditemukan di resume bidang Y
✅ Benchmark kualifikasi yang kompetitif di industri tertentu

PENTING: Jangan sebut nama atau identitas kandidat spesifik.
Fokus pada POLA dan INSIGHT dari data, bukan data individu.

Gunakan bahasa Indonesia yang ramah dan mudah dipahami."""


def resume_agent_node(state: AgentState) -> AgentState:
    """
    Node Resume RAG Agent.
    Prompt berbeda tergantung role: HR (cari kandidat) vs Job Seeker (insight).
    """
    messages  = state["messages"]
    role      = state.get("role", "jobseeker")
    prompt    = PROMPT_HR if role == "hr" else PROMPT_JOBSEEKER

    history         = messages[:-1][-10:]
    current_message = messages[-1]

    react_agent = create_react_agent(llm, ALL_TOOLS)
    result = react_agent.invoke({
        "messages": [SystemMessage(content=prompt)] + history + [current_message]
    })

    final_msg    = result["messages"][-1]
    final_answer = final_msg.content

    # Kumpulkan RAG evidence dari tool outputs
    source_docs = ""
    for msg in result["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            source_docs += msg.content + "\n\n"

    total_tokens = 0
    if hasattr(final_msg, "usage_metadata") and final_msg.usage_metadata:
        total_tokens = final_msg.usage_metadata.get("total_tokens", 0)

    return {
        **state,
        "messages"    : messages + [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "source_docs" : source_docs.strip(),
        "sql_query"   : "",
        "total_tokens": total_tokens,
    }