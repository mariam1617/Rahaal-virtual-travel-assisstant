"""
Tourist Help Advisories & Visa Rules Node (Story 2.3)
Queries the existing TouristRAG pipeline (answer_tourist_query) and prepares
a grounded candidate response for the verification node.
Module: src/agents/tourist_help/advisories.py
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_latest_user_query(state: Dict[str, Any]) -> str:
    """Extract the most recent human message from state."""
    # Prefer the already-resolved user_query if it was freshly set this turn
    q = state.get("user_query", "")
    if q:
        return q
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return str(msg.content).strip()
        if isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]).lower() in {"human", "user"}:
            return str(msg[1]).strip()
    return ""


def _get_nationality(state: Dict[str, Any]) -> Optional[str]:
    """Extract user nationality from preferences if available."""
    prefs = state.get("user_preferences")
    if prefs is None:
        return None
    if hasattr(prefs, "nationality"):
        return prefs.nationality
    if isinstance(prefs, dict):
        return prefs.get("nationality")
    return None


def advisories_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: queries the existing TouristRAG knowledge base for visa,
    transport, safety, and advisory information.

    Uses answer_tourist_query() from src.rag.tourist_rag.answer — the real
    retrieval pipeline — instead of hardcoded static text.

    Passes grounded output to the Help Verification Guardrail Node.
    """
    # Import from the actual module path (src.rag.tourist_rag)
    try:
        from src.rag.tourist_rag.answer import answer_tourist_query
    except ImportError:
        # Fallback: try without the 'tourist_rag' subpackage (alias path)
        from src.rag.answer import answer_tourist_query  # type: ignore

    user_query = _get_latest_user_query(state)
    if not user_query:
        user_query = "Egypt travel advisory"

    nationality = _get_nationality(state)

    logger.debug("[AdvisoriesRAG] query=%r nationality=%r", user_query, nationality)

    try:
        result = answer_tourist_query(
            query=user_query,
            user_nationality=nationality,
            use_llm=True,
        )
    except Exception as exc:
        logger.error("[AdvisoriesRAG] answer_tourist_query failed: %s", exc)
        # Safe fallback — no hardcoded travel facts
        from src.agents.tourist_help.prompts import DEFAULT_ADVISORY_FALLBACK
        return {
            "retrieved_context": "",
            "candidate_response": DEFAULT_ADVISORY_FALLBACK,
            "response": DEFAULT_ADVISORY_FALLBACK,
            "retrieval_sources": [],
            "rag_status": "error",
            "out_of_scope": False,
            "help_domain": "advisories",
            "current_intent": "advisories",
            "next_agent": "advisories_completed",
        }

    # Handle out-of-scope queries (e.g. "visa for France")
    if result.status == "out_of_scope":
        logger.info("[AdvisoriesRAG] out_of_scope for query=%r", user_query)
        return {
            "retrieved_context": "",
            "candidate_response": result.answer,
            "response": result.answer,
            "retrieval_sources": result.sources,
            "rag_status": "out_of_scope",
            "out_of_scope": True,
            "verified_response": result.answer,
            "is_verified": True,                # skip verification for scope refusals
            "help_domain": "advisories",
            "current_intent": "advisories",
            "next_agent": "advisories_completed",
        }

    # Handle no-answer (KB has no relevant data)
    if result.status == "no_answer":
        logger.info("[AdvisoriesRAG] no_answer for query=%r", user_query)
        return {
            "retrieved_context": result.context,
            "candidate_response": result.answer,
            "response": result.answer,
            "retrieval_sources": result.sources,
            "rag_status": "no_answer",
            "out_of_scope": False,
            "help_domain": "advisories",
            "current_intent": "advisories",
            "next_agent": "advisories_completed",
        }

    # Successful retrieval
    logger.debug("[AdvisoriesRAG] answered. context_len=%d", len(result.context))
    return {
        "retrieved_context": result.context,
        "candidate_response": result.answer,
        "response": result.answer,
        "retrieval_sources": result.sources,
        "rag_status": "answered",
        "out_of_scope": False,
        "pending_question": result.pending_question,
        "help_domain": "advisories",
        "current_intent": "advisories",
        "next_agent": "advisories_completed",
    }


# Backward-compatible alias
advisories_agent_node = advisories_rag_node
