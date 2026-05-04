"""
graph/workflow.py
-----------------
Menyusun StateGraph Langgraph dengan tiga agent.

Alur:
  START
    │
    ▼
  supervisor ──────────────┬──> sql_agent    (HR only) ──> END
                            ├──> resume_agent            ──> END
                            └──> general_agent           ──> END
"""

from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.sql_agent import sql_agent_node
from agents.resume_agent import resume_agent_node
from agents.general_agent import general_agent_node


def route_after_supervisor(
    state: AgentState,
) -> Literal["sql_agent", "resume_agent", "general_agent"]:
    """Conditional edge berdasarkan keputusan supervisor."""
    return state["next_agent"]


def build_graph():
    """Merakit dan compile StateGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor",    supervisor_node)
    workflow.add_node("sql_agent",     sql_agent_node)
    workflow.add_node("resume_agent",  resume_agent_node)
    workflow.add_node("general_agent", general_agent_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "sql_agent"    : "sql_agent",
            "resume_agent" : "resume_agent",
            "general_agent": "general_agent",
        },
    )
    workflow.add_edge("sql_agent",     END)
    workflow.add_edge("resume_agent",  END)
    workflow.add_edge("general_agent", END)

    return workflow.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(user_input: str, chat_history: list, role: str) -> dict:
    """
    Entry point utama dari Streamlit.

    Args:
        user_input  : Pesan terbaru dari user.
        chat_history: List of BaseMessage dari sesi sebelumnya.
        role        : "hr" atau "jobseeker".

    Returns:
        dict: answer, source_docs, sql_query, total_tokens, next_agent
    """
    graph = get_graph()

    initial_state: AgentState = {
        "messages"    : chat_history[-10:] + [HumanMessage(content=user_input)],
        "role"        : role,
        "next_agent"  : "",
        "final_answer": "",
        "source_docs" : "",
        "sql_query"   : "",
        "total_tokens": 0,
    }

    result = graph.invoke(initial_state)

    return {
        "answer"      : result["final_answer"],
        "source_docs" : result["source_docs"],
        "sql_query"   : result["sql_query"],
        "total_tokens": result["total_tokens"],
        "next_agent"  : result["next_agent"],
    }