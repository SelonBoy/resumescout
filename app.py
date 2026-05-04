"""
app.py
------
Entry point Streamlit — ResumeScout AI-Powered Recruitment Platform.

Flow:
  1. Cek session state → sudah login?
  2. Jika belum → render login page
  3. Jika sudah → render chatbot sesuai role (hr / jobseeker)

Fitur:
  ✅ Login dengan autentikasi MySQL
  ✅ Role-based UI (HR vs Job Seeker)
  ✅ Multi-agent: Supervisor → SQL / RAG / General Agent
  ✅ Chat history (5 turn terakhir)
  ✅ RAG Evidence display (dokumen dari Qdrant)
  ✅ SQL Query display (query yang dieksekusi)
  ✅ Agent badge (🗄 SQL / 🔍 RAG / 💬 General)
  ✅ Token usage tracker
  ✅ Logout button
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from auth.login import render_login_page
from graph.workflow import run_agent

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeScout",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge-sql     { background:#e67e22; color:#fff; padding:2px 10px; border-radius:12px; font-size:.75em; font-weight:600; display:inline-block; margin-top:6px; }
.badge-rag     { background:#1a6cc4; color:#fff; padding:2px 10px; border-radius:12px; font-size:.75em; font-weight:600; display:inline-block; margin-top:6px; }
.badge-general { background:#1e8a4a; color:#fff; padding:2px 10px; border-radius:12px; font-size:.75em; font-weight:600; display:inline-block; margin-top:6px; }
.token-info    { color:#999; font-size:.78em; font-style:italic; margin-top:4px; }
.source-box    { background:#f5f8ff; border-left:4px solid #1a6cc4; padding:12px 14px; border-radius:4px; font-size:.82em; white-space:pre-wrap; font-family:monospace; }
.sql-box       { background:#fff8f0; border-left:4px solid #e67e22; padding:12px 14px; border-radius:4px; font-size:.85em; font-family:monospace; }
.role-badge-hr { background:#c0392b; color:#fff; padding:3px 12px; border-radius:12px; font-size:.8em; font-weight:600; }
.role-badge-js { background:#2980b9; color:#fff; padding:3px 12px; border-radius:12px; font-size:.8em; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "user"              : None,
        "username"          : "",
        "role"              : "",
        "messages"          : [],
        "lc_history"        : [],
        "metadata"          : [],
        "total_tokens_used" : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Login Gate ────────────────────────────────────────────────────────────────
if not st.session_state["user"]:
    render_login_page()
    st.stop()


# ── Logged In — Setup Vars ────────────────────────────────────────────────────
role     = st.session_state["role"]
username = st.session_state["username"]
is_hr    = role == "hr"

EXAMPLES_HR = [
    "Berapa jumlah kandidat per kategori?",
    "Tampilkan semua kategori yang tersedia",
    "Ada berapa total kandidat di database?",
    "Carikan software engineer yang berpengalaman Python",
    "Siapa kandidat terbaik untuk posisi data scientist?",
]

EXAMPLES_JS = [
    "Skill apa yang umum dimiliki engineer di database?",
    "Pengalaman seperti apa yang ada di resume Finance?",
    "Insight dari resume bidang Healthcare?",
    "Tips membuat resume yang ATS-friendly",
    "Bagaimana cara mempersiapkan wawancara kerja?",
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 ResumeScout")
    st.caption("AI-Powered Recruitment Platform")
    st.divider()

    # User info
    role_html = (
        '<span class="role-badge-hr">👔 HR / Rekruter</span>'
        if is_hr else
        '<span class="role-badge-js">🎯 Job Seeker</span>'
    )
    st.markdown(f"**Login sebagai:** {username}")
    st.markdown(role_html, unsafe_allow_html=True)
    st.divider()

    # Session stats
    st.markdown("#### 📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Token Used", f"{st.session_state.total_tokens_used:,}")
    c2.metric("Pesan", len(st.session_state.messages))

    if is_hr:
        st.markdown("**Agent tersedia:**")
        st.markdown("🗄 SQL Agent · 🔍 RAG Agent · 💬 General")
    else:
        st.markdown("**Agent tersedia:**")
        st.markdown("🔍 RAG Agent · 💬 General Agent")

    st.divider()

    # Kategori (hanya tampil untuk HR)
    if is_hr:
        st.markdown("#### 🗂 Kategori Database")
        CATEGORIES = [
            "HR", "DESIGNER", "INFORMATION-TECHNOLOGY", "TEACHER",
            "ADVOCATE", "BUSINESS-DEVELOPMENT", "HEALTHCARE", "FITNESS",
            "AGRICULTURE", "BPO", "SALES", "CONSULTANT", "DIGITAL-MEDIA",
            "AUTOMOBILE", "CHEF", "FINANCE", "APPAREL", "ENGINEERING",
            "ACCOUNTANT", "CONSTRUCTION", "PUBLIC-RELATIONS",
            "BANKING", "ARTS", "AVIATION",
        ]
        st.code(" · ".join(CATEGORIES), language=None)
        st.divider()

    # Contoh pertanyaan
    st.markdown("#### 💡 Contoh Pertanyaan")
    examples = EXAMPLES_HR if is_hr else EXAMPLES_JS
    for q in examples:
        if st.button(q, use_container_width=True, key=f"ex_{q[:15]}"):
            st.session_state["queued_input"] = q
    st.divider()

    # Logout + Clear
    col_a, col_b = st.columns(2)
    if col_a.button("🗑 Clear", use_container_width=True):
        for k in ["messages", "lc_history", "metadata"]:
            st.session_state[k] = []
        st.session_state["total_tokens_used"] = 0
        st.rerun()
    if col_b.button("🚪 Logout", use_container_width=True, type="secondary"):
        for k in ["user", "username", "role", "messages", "lc_history", "metadata", "total_tokens_used"]:
            st.session_state[k] = None if k == "user" else ([] if isinstance(st.session_state[k], list) else ("" if isinstance(st.session_state[k], str) else 0))
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
if is_hr:
    st.markdown("## 👔 ResumeScout — HR Dashboard")
    st.markdown(
        "Cari kandidat dari **2.484 resume** menggunakan AI. "
        "Pertanyaan analitik dijawab via **SQL Agent** (MySQL), "
        "pencarian kandidat via **RAG Agent** (Qdrant)."
    )
else:
    st.markdown("## 🎯 ResumeScout — Career Insights")
    st.markdown(
        "Dapatkan insight dari **2.484 resume nyata** untuk membantu perjalanan karirmu. "
        "Tanya tentang skill umum, benchmark industri, dan tips karir."
    )
st.divider()


# ── Render Chat History ───────────────────────────────────────────────────────
meta_idx = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and meta_idx < len(st.session_state.metadata):
            meta      = st.session_state.metadata[meta_idx]
            meta_idx += 1

            agent = meta.get("next_agent", "")
            if agent == "sql_agent":
                st.markdown('<span class="badge-sql">🗄 SQL Agent</span>', unsafe_allow_html=True)
            elif agent == "resume_agent":
                st.markdown('<span class="badge-rag">🔍 RAG Agent</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-general">💬 General Agent</span>', unsafe_allow_html=True)

            t = meta.get("total_tokens", 0)
            if t:
                st.markdown(f'<p class="token-info">🪙 {t:,} tokens digunakan</p>', unsafe_allow_html=True)

            # RAG Evidence
            docs = meta.get("source_docs", "")
            if docs:
                with st.expander("📚 RAG Evidence — Dokumen dari Qdrant", expanded=False):
                    st.markdown(f'<div class="source-box">{docs}</div>', unsafe_allow_html=True)

            # SQL Query
            sql = meta.get("sql_query", "")
            if sql:
                with st.expander("🗄 SQL Query yang Dieksekusi", expanded=False):
                    st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)


# ── Chat Input ────────────────────────────────────────────────────────────────
placeholder = (
    "Tanya tentang kandidat atau statistik database..."
    if is_hr else
    "Tanya tentang skill, benchmark industri, atau tips karir..."
)

queued     = st.session_state.pop("queued_input", None)
user_input = st.chat_input(placeholder) or queued

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Sedang memproses..."):
            result = run_agent(
                user_input=user_input,
                chat_history=st.session_state.lc_history,
                role=role,
            )

        answer       = result["answer"]
        source_docs  = result["source_docs"]
        sql_query    = result["sql_query"]
        total_tokens = result["total_tokens"]
        next_agent   = result["next_agent"]

        st.markdown(answer)

        # Badge
        if next_agent == "sql_agent":
            st.markdown('<span class="badge-sql">🗄 SQL Agent</span>', unsafe_allow_html=True)
        elif next_agent == "resume_agent":
            st.markdown('<span class="badge-rag">🔍 RAG Agent</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-general">💬 General Agent</span>', unsafe_allow_html=True)

        if total_tokens:
            st.markdown(f'<p class="token-info">🪙 {total_tokens:,} tokens digunakan</p>', unsafe_allow_html=True)

        if source_docs:
            with st.expander("📚 RAG Evidence — Dokumen dari Qdrant", expanded=True):
                st.markdown(f'<div class="source-box">{source_docs}</div>', unsafe_allow_html=True)

        if sql_query:
            with st.expander("🗄 SQL Query yang Dieksekusi", expanded=True):
                st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)

    # Simpan ke session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.metadata.append({
        "source_docs" : source_docs,
        "sql_query"   : sql_query,
        "total_tokens": total_tokens,
        "next_agent"  : next_agent,
    })
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=answer))
    st.session_state.total_tokens_used += total_tokens

    st.rerun()