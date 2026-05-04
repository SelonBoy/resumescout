"""
graph/state.py
--------------
AgentState yang dibawa sepanjang alur Langgraph.
Field 'role' dipakai supervisor untuk routing yang role-aware.
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages    : Annotated[List[BaseMessage], add_messages]
    role        : str   # "hr" atau "jobseeker"
    next_agent  : str   # "sql_agent" | "resume_agent" | "general_agent"
    final_answer: str
    source_docs : str   # RAG evidence dari Qdrant (kosong jika SQL/General)
    sql_query   : str   # Query SQL yang dieksekusi (kosong jika bukan SQL agent)
    total_tokens: int