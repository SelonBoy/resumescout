"""
rag/tools.py
------------
RAG Tools yang dieksekusi oleh Resume RAG Agent.

Tools:
  1. semantic_search_tool  — pencarian bebas berdasarkan deskripsi
  2. category_search_tool  — pencarian + filter kategori di Qdrant
"""

from typing import List
from langchain.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.retriever import embed_query, search, format_hits

VALID_CATEGORIES: List[str] = [
    "HR", "DESIGNER", "INFORMATION-TECHNOLOGY", "TEACHER", "ADVOCATE",
    "BUSINESS-DEVELOPMENT", "HEALTHCARE", "FITNESS", "AGRICULTURE", "BPO",
    "SALES", "CONSULTANT", "DIGITAL-MEDIA", "AUTOMOBILE", "CHEF", "FINANCE",
    "APPAREL", "ENGINEERING", "ACCOUNTANT", "CONSTRUCTION",
    "PUBLIC-RELATIONS", "BANKING", "ARTS", "AVIATION",
]


@tool
def semantic_search_tool(query: str) -> str:
    """
    Cari resume yang paling relevan menggunakan semantic similarity di Qdrant.
    Gunakan ini ketika user tidak menyebut kategori spesifik, atau ketika
    job seeker ingin melihat insight skill umum dari suatu bidang.

    Args:
        query: Deskripsi kandidat atau skill yang dicari.

    Returns:
        Daftar resume relevan beserta kategori dan kontennya.
    """
    vector = embed_query(query)
    hits = search(query_vector=vector)
    return format_hits(hits)


@tool
def category_search_tool(query: str, category: str) -> str:
    """
    Cari resume dalam kategori pekerjaan tertentu dengan metadata filter Qdrant.
    Gunakan ini ketika user menyebut bidang pekerjaan yang spesifik.

    Args:
        query   : Deskripsi skill atau kualifikasi yang dicari.
        category: Kategori pekerjaan (HURUF KAPITAL). Pilihan valid:
                  HR, DESIGNER, INFORMATION-TECHNOLOGY, TEACHER, ADVOCATE,
                  BUSINESS-DEVELOPMENT, HEALTHCARE, FITNESS, AGRICULTURE,
                  BPO, SALES, CONSULTANT, DIGITAL-MEDIA, AUTOMOBILE, CHEF,
                  FINANCE, APPAREL, ENGINEERING, ACCOUNTANT, CONSTRUCTION,
                  PUBLIC-RELATIONS, BANKING, ARTS, AVIATION

    Returns:
        Daftar resume dalam kategori tersebut yang paling relevan.
    """
    category_upper = category.upper().strip()
    if category_upper not in VALID_CATEGORIES:
        closest = [c for c in VALID_CATEGORIES if category_upper[:4] in c]
        suggestion = f" Mungkin maksudnya: {', '.join(closest)}?" if closest else ""
        return (
            f"Kategori '{category}' tidak ditemukan.{suggestion}\n"
            f"Kategori valid: {', '.join(VALID_CATEGORIES)}"
        )

    vector = embed_query(query)
    hits = search(
        query_vector=vector,
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category_upper))]
        ),
    )
    return format_hits(hits)


ALL_TOOLS: List = [semantic_search_tool, category_search_tool]