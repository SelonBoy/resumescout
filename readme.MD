# 📄 ResumeScout — AI-Powered Recruitment Platform

Chatbot multi-agent berbasis **RAG + Text-to-SQL** untuk platform rekrutmen.
Database kandidat dari **2.484 resume nyata** di 24 bidang pekerjaan.

## 🏗 Arsitektur

```
User Login (MySQL Auth)
        │
        ▼
  SUPERVISOR AGENT  ← role-aware routing
        │
   ┌────┼────────┐
   ▼    ▼        ▼
 SQL   RAG    General
Agent  Agent   Agent
  │      │
MySQL  Qdrant
(kandidat) (embeddings)
```

**Tiga agent:**
- `sql_agent`     → Text-to-SQL ke MySQL `candidates` table (HR only)
- `resume_agent`  → Semantic search ke Qdrant (HR: cari kandidat | JS: insight)
- `general_agent` → Career advisor tanpa tools

## 📁 Struktur Project

```
resume-rag-chatbot/
├── app.py
├── auth/
│   ├── __init__.py
│   └── login.py          ← bcrypt auth via MySQL
├── agents/
│   ├── __init__.py
│   ├── supervisor.py     ← role-aware routing
│   ├── sql_agent.py      ← Text-to-SQL (HR only)
│   ├── resume_agent.py   ← RAG agent (Qdrant)
│   └── general_agent.py  ← Career advisor
├── rag/
│   ├── __init__.py
│   ├── retriever.py      ← Qdrant + embeddings
│   └── tools.py          ← semantic + category search tools
├── db/
│   ├── __init__.py
│   └── connection.py     ← SQLAlchemy + LangChain SQLDatabase
├── graph/
│   ├── __init__.py
│   ├── state.py          ← AgentState TypedDict
│   └── workflow.py       ← StateGraph build + run_agent()
├── scripts/
│   └── ingest.py         ← Full setup: MySQL + Qdrant
├── data/
│   └── Resume.csv
├── .env.example
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Isi environment variables
```bash
cp .env.example .env
# Edit .env
```

### 3. Jalankan setup (SEKALI)
```bash
python scripts/ingest.py
```
Script ini akan:
- Buat database & tabel MySQL
- Insert default users
- Preprocessing & insert CSV ke MySQL
- Embed + upload ke Qdrant

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

## 🔐 Demo Accounts

| Role | Username | Password |
|------|----------|----------|
| HR / Rekruter | `hr_user` | `hr123` |
| Job Seeker | `jobseeker` | `js123` |

## 🗄 MySQL Schema

```sql
-- Autentikasi user
CREATE TABLE users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(50) UNIQUE,
    password_hash VARCHAR(255),
    role          ENUM('hr', 'jobseeker')
);

-- Data kandidat dari Resume.csv
CREATE TABLE candidates (
    id        INT PRIMARY KEY,
    category  VARCHAR(50),
    resume_id INT,
    INDEX idx_category (category)
);
```