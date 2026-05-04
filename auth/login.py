"""
auth/login.py
-------------
Autentikasi user menggunakan MySQL.
Password di-hash dengan bcrypt sebelum disimpan.

Dua role:
  - hr        : bisa akses SQL Agent + RAG Agent + General Agent
  - jobseeker : bisa akses RAG Agent (insight mode) + General Agent
"""

import bcrypt
import streamlit as st
from sqlalchemy import text
from db.connection import get_engine
from typing import Optional

# Core Auth Functions

def hash_password(plain: str) -> str:
    """Hash password menggunakan bcrypt."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verifikasi password terhadap hash yang tersimpan di DB."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def authenticate(username: str, password: str) -> Optional[dict]:
    """
    Cek username + password ke tabel users di MySQL.

    Returns:
        dict {"username": str, "role": str} jika valid.
        None jika username/password salah.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT password_hash, role FROM users WHERE username = :u"),
            {"u": username},
        ).fetchone()

    if result is None:
        return None  # username tidak ditemukan

    hashed, role = result
    if not verify_password(password, hashed):
        return None  # password salah

    return {"username": username, "role": role}


# ── Streamlit Login Page ──────────────────────────────────────────────────────

def render_login_page():
    """
    Render halaman login Streamlit.
    Menyimpan user info ke session state jika berhasil login.
    """
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## 📄 ResumeScout")
        st.markdown("#### AI-Powered Recruitment Platform")
        st.divider()

        st.markdown("### 🔐 Login")
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")

        if st.button("Login", use_container_width=True, type="primary"):
            if not username or not password:
                st.warning("Username dan password tidak boleh kosong.")
                return

            with st.spinner("Memverifikasi..."):
                user = authenticate(username, password)

            if user:
                st.session_state["user"]     = user
                st.session_state["username"] = user["username"]
                st.session_state["role"]     = user["role"]
                st.success(f"Login berhasil! Selamat datang, {username} 👋")
                st.rerun()
            else:
                st.error("Username atau password salah.")

        st.divider()
        st.caption("Demo accounts — **HR**: `hr_user` / `hr123`  |  **Job Seeker**: `jobseeker` / `js123`")