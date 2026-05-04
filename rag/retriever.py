"""
rag/retriever.py
----------------
Inisialisasi Qdrant client dan OpenAI Embeddings (singleton).
Semua operasi vector search dikelola di sini.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")
TOP_K = 5


@lru_cache(maxsize=1)
def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def embed_query(query: str) -> List[float]:
    return get_embedder().embed_query(query)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed_documents(texts)


def search(query_vector: List[float], limit: int = TOP_K, query_filter=None) -> list:
    return get_qdrant_client().search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
    )


def format_hits(hits: list) -> str:
    """Format Qdrant hits menjadi teks yang mudah dibaca LLM."""
    if not hits:
        return "Tidak ditemukan resume yang relevan di database."
    parts = []
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        parts.append(
            f"[Resume {i}] (score: {round(hit.score, 3)})\n"
            f"Category  : {p.get('category', 'N/A')}\n"
            f"Resume ID : {p.get('resume_id', 'N/A')}\n"
            f"Content   :\n{p.get('text', '').strip()}\n"
        )
    return "\n---\n".join(parts)