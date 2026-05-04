"""
agents/sql_agent.py
-------------------
SQL Agent — menggunakan LangChain create_sql_agent untuk Text-to-SQL.
Hanya bisa diakses oleh HR role.

Agent ini otomatis:
  1. Membaca schema tabel 'candidates' di MySQL
  2. Mengubah pertanyaan natural language → SQL query
  3. Mengeksekusi query dan mengembalikan hasil
  4. Merangkum hasil dalam bahasa natural

Table yang bisa diakses: candidates (id, category, resume_id)
Table yang tidak terekspos: users (keamanan)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from db.connection import get_langchain_db
from graph.state import AgentState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

SQL_AGENT_PREFIX = """Kamu adalah Database Analyst yang membantu HR mendapatkan insight dari database kandidat.

Kamu memiliki akses ke tabel MySQL berikut:
- **candidates**: berisi data kandidat dengan kolom id, category, resume_id

Panduan:
✅ Tulis query SQL yang efisien dan akurat
✅ Selalu tampilkan hasil dengan format yang mudah dibaca
✅ Jika diminta jumlah/statistik, gunakan COUNT, GROUP BY, dll
✅ Jelaskan hasil query dalam bahasa Indonesia yang ramah
✅ Jika query tidak menghasilkan data, sampaikan dengan jelas

Jangan pernah mengakses atau menyebut tabel selain 'candidates'."""


def sql_agent_node(state: AgentState) -> AgentState:
    """
    Node SQL Agent.
    Menggunakan LangChain create_sql_agent untuk Text-to-SQL ke MySQL.
    Hanya dipanggil untuk user dengan role 'hr'.
    """
    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
        messages[-1].content,
    )

    # Build SQL agent dengan toolkit
    db = get_langchain_db()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        prefix=SQL_AGENT_PREFIX,
        verbose=False,
    )

    try:
        result = agent.invoke({"input": last_human})
        final_answer = result.get("output", "Tidak bisa mendapatkan hasil dari database.")
    except Exception as e:
        final_answer = f"Terjadi error saat query database: {str(e)}"

    # Coba ekstrak SQL query yang dieksekusi (untuk ditampilkan di UI)
    sql_query = ""
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if isinstance(step, tuple) and len(step) > 0:
                action = step[0]
                if hasattr(action, "tool") and "sql" in action.tool.lower():
                    sql_query = getattr(action, "tool_input", {}).get("query", "")
                    break

    return {
        **state,
        "messages"    : messages + [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "source_docs" : "",
        "sql_query"   : sql_query,
        "total_tokens": 0,
    }