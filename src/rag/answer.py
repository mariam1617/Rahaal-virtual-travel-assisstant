# Re-export shim: src/rag/answer.py
# The RAG modules were placed in src/rag/tourist_rag/ but import each other
# using the shorter src.rag.* path. This shim bridges the two.
from src.rag.tourist_rag.answer import (  # noqa: F401
    RAGResult,
    answer_tourist_query,
    answer_emergency_query,
    llm_answer,
    extractive_answer,
    get_answer_llm,
    MAX_ANSWER_CHARS,
    MAX_UNITS,
    VISA_PORTAL,
    ANSWER_SYSTEM_PROMPT,
    CORE_EMERGENCY_NUMBERS,
)
