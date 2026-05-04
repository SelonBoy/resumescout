"""
db/connection.py
----------------
Koneksi ke MySQL menggunakan SQLAlchemy.
Menyediakan engine untuk raw queries dan LangChain SQLDatabase wrapper
yang dipakai oleh SQL Agent.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from langchain_community.utilities import SQLDatabase

load_dotenv()


def _build_url() -> str:
    """Build MySQL connection URL dari environment variables."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL belum diset!")

    return url

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Singleton SQLAlchemy engine.
    Dipakai untuk query langsung (auth, insert, dll).
    """
    url = _build_url()
    return create_engine(url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_langchain_db() -> SQLDatabase:
    """
    Singleton LangChain SQLDatabase wrapper.
    Dipakai oleh SQL Agent untuk Text-to-SQL.
    Hanya expose table 'candidates' agar agent tidak bisa akses tabel users.
    """
    url = _build_url()
    return SQLDatabase.from_uri(
        url,
        include_tables=["candidates"],
        sample_rows_in_table_info=3,
    )