"""
agents/supervisor.py
--------------------
Supervisor Agent — routing pertanyaan ke agent yang tepat
berdasarkan INTENT user DAN ROLE yang sedang login.

Routing rules:
  HR role:
    - Pertanyaan analitik/statistik → sql_agent
    - Pencarian kandidat spesifik   → resume_agent
    - Pertanyaan umum karir         → general_agent

  Job Seeker role:
    - Pertanyaan tentang insight/skill dari bidang tertentu → resume_agent
    - Pertanyaan umum karir, tips resume                   → general_agent
    - Pertanyaan analitik (statistik DB)                   → general_agent
      (job seeker tidak perlu akses SQL langsung)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

SUPERVISOR_PROMPT_HR = """Kamu adalah supervisor yang mengarahkan pertanyaan HR/rekruter ke agent yang tepat.

Tiga pilihan agent:

1. **sql_agent** — Pilih jika pertanyaan bersifat ANALITIK atau STATISTIK:
   - Berapa jumlah kandidat per kategori?
   - Kategori apa saja yang tersedia?
   - Tampilkan total resume di database
   - Ada berapa kandidat IT?

2. **resume_agent** — Pilih jika HR ingin MENCARI KANDIDAT SPESIFIK:
   - Carikan engineer yang bisa Python
   - Siapa kandidat cocok untuk posisi data scientist?
   - Tampilkan resume Finance yang berpengalaman
   - Bandingkan kandidat di bidang healthcare

3. **general_agent** — Pilih jika pertanyaan UMUM tentang karir/rekrutmen:
   - Tips wawancara yang baik
   - Apa bedanya CV dan resume?
   - Salam atau basa-basi

Perhatikan konteks percakapan sebelumnya.
Jawab HANYA satu kata: sql_agent / resume_agent / general_agent"""

SUPERVISOR_PROMPT_JOBSEEKER = """Kamu adalah supervisor yang mengarahkan pertanyaan job seeker ke agent yang tepat.

Dua pilihan agent:

1. **resume_agent** — Pilih jika job seeker ingin INSIGHT dari database resume:
   - Skill apa yang umum di resume engineer?
   - Pengalaman seperti apa yang banyak dimiliki kandidat Finance?
   - Contoh resume dari bidang IT seperti apa?

2. **general_agent** — Pilih untuk semua pertanyaan UMUM:
   - Tips membuat resume yang bagus
   - Cara mempersiapkan wawancara kerja
   - Salam atau pertanyaan umum seputar karir
   - Pertanyaan statistik database (arahkan ke sini, bukan ke SQL)

Perhatikan konteks percakapan sebelumnya.
Jawab HANYA satu kata: resume_agent / general_agent"""


def supervisor_node(state: AgentState) -> AgentState:
    """
    Node supervisor: membaca role user + intent pertanyaan,
    lalu memutuskan agent yang paling tepat.
    """
    messages = state["messages"]
    role = state.get("role", "jobseeker")

    # Pilih prompt berdasarkan role
    system_prompt = SUPERVISOR_PROMPT_HR if role == "hr" else SUPERVISOR_PROMPT_JOBSEEKER

    # Ambil 6 pesan terakhir sebagai konteks
    history_str = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages[-6:]
    )
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Role user: {role}\n"
                f"Chat history:\n{history_str}\n\n"
                f"Pertanyaan terbaru: {last_human}\n\n"
                f"Routing ke agent mana?"
            )
        ),
    ])

    decision = response.content.strip().lower()

    if "sql" in decision and role == "hr":
        next_agent = "sql_agent"
    elif "resume" in decision:
        next_agent = "resume_agent"
    else:
        next_agent = "general_agent"

    return {**state, "next_agent": next_agent}