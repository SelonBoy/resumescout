"""
scripts/ingest.py
-----------------
Script setup lengkap — jalankan SEKALI sebelum pertama kali pakai aplikasi.

Yang dilakukan script ini:
  1. Buat database & tabel MySQL (users + candidates)
  2. Insert default users (hr_user & jobseeker)
  3. Preprocessing Resume.csv (drop kolom tidak perlu, clean teks)
  4. Insert data candidates ke MySQL
  5. Embed Resume_str → upload ke Qdrant Cloud

Usage:
    python scripts/ingest.py

Pastikan .env sudah terisi dan MySQL server berjalan.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag.retriever import get_embedder

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
from urllib.parse import quote_plus

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL belum diset!")

engine = create_engine(DATABASE_URL)
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")
CSV_PATH        = os.path.join(os.path.dirname(__file__), "..", "data", "Resume.csv")
VECTOR_SIZE     = 1536
BATCH_SIZE      = 10
MAX_CHARS       = 2000
# ─────────────────────────────────────────────────────────────────────────────


def get_engine():
    return create_engine(os.getenv("DATABASE_URL"))


# ── Step 1: Setup MySQL ───────────────────────────────────────────────────────
def setup_mysql():
    print("\n[1/5] Setup MySQL...")

    engine = get_engine()

    with engine.connect() as conn:
        # users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('hr', 'jobseeker') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # candidates table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INT PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                resume_id INT NOT NULL,
                INDEX idx_category (category)
            )
        """))

        conn.commit()

    print("  ✅ Tabel 'users' dan 'candidates' siap.")


# ── Step 2: Insert Default Users ─────────────────────────────────────────────
def insert_default_users():
    print("\n[2/5] Insert default users...")

    default_users = [
        ("hr_user",   "hr123",  "hr"),
        ("jobseeker", "js123",  "jobseeker"),
    ]

    engine = get_engine()
    with engine.connect() as conn:
        for username, plain_pw, role in default_users:
            # Cek apakah sudah ada
            existing = conn.execute(
                text("SELECT id FROM users WHERE username = :u"),
                {"u": username}
            ).fetchone()

            if existing:
                print(f"  ⚠️  User '{username}' sudah ada, skip.")
                continue

            pw_hash = bcrypt.hashpw(plain_pw.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                text("INSERT INTO users (username, password_hash, role) VALUES (:u, :p, :r)"),
                {"u": username, "p": pw_hash, "r": role},
            )
            print(f"  ✅ User '{username}' ({role}) berhasil dibuat.")
        conn.commit()


# ── Step 3: Preprocessing CSV ────────────────────────────────────────────────
def preprocess_csv() -> pd.DataFrame:
    print("\n[3/5] Preprocessing Resume.csv...")

    df = pd.read_csv(CSV_PATH)
    print(f"  Raw data: {len(df):,} rows | Columns: {df.columns.tolist()}")

    # Drop kolom yang tidak diperlukan
    df = df[["ID", "Resume_str", "Category"]].dropna()

    # Clean teks
    df["Resume_str"] = (
        df["Resume_str"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)   # hapus whitespace berlebih
        .str[:MAX_CHARS]                          # truncate untuk hemat token
    )

    # Normalisasi kategori ke uppercase
    df["Category"] = df["Category"].str.upper().str.strip()

    print(f"  ✅ After cleaning: {len(df):,} rows | {df['Category'].nunique()} categories")
    print(f"  Kategori: {sorted(df['Category'].unique().tolist())}")
    return df


# ── Step 4: Insert ke MySQL ───────────────────────────────────────────────────
def insert_to_mysql(df: pd.DataFrame):
    print("\n[4/5] Insert candidates ke MySQL...")

    engine = get_engine()
    with engine.connect() as conn:
        # Cek apakah sudah ada data
        count = conn.execute(text("SELECT COUNT(*) FROM candidates")).scalar()
        if count > 0:
            print(f"  ⚠️  Tabel candidates sudah berisi {count:,} rows, skip insert.")
            return

        inserted = 0
        for _, row in df.iterrows():
            conn.execute(
                text("INSERT IGNORE INTO candidates (id, category, resume_id) VALUES (:id, :cat, :rid)"),
                {"id": int(row["ID"]), "cat": row["Category"], "rid": int(row["ID"])},
            )
            inserted += 1
        conn.commit()
    print(f"  ✅ {inserted:,} candidates berhasil di-insert ke MySQL.")


# ── Step 5: Embed & Upload ke Qdrant ─────────────────────────────────────────
def upload_to_qdrant(df: pd.DataFrame):
    print("\n[5/5] Embed & upload ke Qdrant...")

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = get_embedder()

    # Buat collection jika belum ada
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  ⚠️  Collection '{COLLECTION_NAME}' sudah ada, skip pembuatan.")
    else:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  ✅ Collection '{COLLECTION_NAME}' berhasil dibuat.")

    total    = len(df)
    uploaded = 0

    for start in range(0, total, BATCH_SIZE):
        batch   = df.iloc[start : start + BATCH_SIZE]
        texts   = batch["Resume_str"].tolist()
        ids     = batch["ID"].tolist()
        cats    = batch["Category"].tolist()

        vectors = embedder.embed_documents(texts)
        points  = [
            PointStruct(
                id=int(ids[i]),
                vector=vectors[i],
                payload={
                    "resume_id": int(ids[i]),
                    "category" : cats[i],
                    "text"     : texts[i],
                },
            )
            for i in range(len(batch))
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += len(batch)
        print(f"  [{(uploaded/total*100):5.1f}%] {uploaded:,}/{total:,} uploaded...")

    print(f"  ✅ {uploaded:,} resumes berhasil masuk ke Qdrant.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ResumeScout — Full Setup (MySQL + Qdrant)")
    print("=" * 60)

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ QDRANT_URL dan QDRANT_API_KEY harus diisi di .env!")
        sys.exit(1)

    setup_mysql()
    insert_default_users()
    df = preprocess_csv()
    insert_to_mysql(df)
    upload_to_qdrant(df)

    print("\n" + "=" * 60)
    print("  ✅ Setup selesai! Siap jalankan: streamlit run app.py")
    print("  Demo login:")
    print("    HR        → username: hr_user   | password: hr123")
    print("    Jobseeker → username: jobseeker | password: js123")
    print("=" * 60)


if __name__ == "__main__":
    main()
