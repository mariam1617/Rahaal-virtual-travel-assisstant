"""
Tourist Help Recommendations Node (Story 2.4)
Provides local attractions and information filtered by the requested destination
and user preferences. Uses the existing TouristRAG knowledge base.
Module: src/agents/tourist_help/recommendations.py
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Egypt cities/regions that can be destination targets
_EGYPT_DESTINATIONS = [
    "luxor", "aswan", "alexandria", "hurghada", "sharm el-sheikh", "sharm",
    "siwa", "dahab", "sinai", "marsa alam", "el gouna", "red sea",
    "fayoum", "bahariya", "abu simbel", "saint catherine", "nuweiba", "taba",
    "marsa matruh", "safaga", "el alamein", "suez", "ismailia", "port said",
    "cairo", "giza",
]

_DESTINATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in sorted(_EGYPT_DESTINATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _get_latest_user_query(state: Dict[str, Any]) -> str:
    """Extract the most recent human message from state."""
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


def _extract_requested_destination(query: str) -> Optional[str]:
    """
    Extract the destination explicitly mentioned in the user's query.
    Returns None if no recognized Egypt destination is found.
    Priority: destination stated in query > anything else.
    """
    m = _DESTINATION_RE.search(query)
    return m.group(1).title() if m else None


def _get_user_interests(state: Dict[str, Any]) -> List[str]:
    """Extract user interests from preferences if available."""
    prefs = state.get("user_preferences")
    if prefs is None:
        return []
    if hasattr(prefs, "interests"):
        return list(prefs.interests or [])
    if isinstance(prefs, dict):
        return list(prefs.get("interests", []))
    return []


def _build_enriched_query(user_query: str, destination: Optional[str], interests: List[str]) -> str:
    """
    Build a richer query string for the knowledge-base search.
    Incorporates the destination and relevant user interests.
    """
    parts = [user_query]
    # Append interest terms only if not already in the query
    q_lower = user_query.lower()
    for interest in interests[:3]:  # limit to avoid query dilution
        if interest.lower() not in q_lower:
            parts.append(interest)
    return " ".join(parts)


def recommendations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: retrieves place recommendations from the existing TouristRAG
    knowledge base.

    Destination priority (highest to lowest):
      1. City/region explicitly mentioned in the user query.
      2. current_location (if no destination stated in query).
    
    Never fabricates attraction names, prices, or ratings.
    If no KB data is found for the requested destination, returns a graceful message.
    """
    # Import from the actual module path
    try:
        from src.rag.tourist_rag.answer import answer_tourist_query
    except ImportError:
        from src.rag.answer import answer_tourist_query  # type: ignore

    user_query = _get_latest_user_query(state)
    if not user_query:
        user_query = "places to visit in Egypt"

    # 1. Determine the requested destination from the query first
    destination = _extract_requested_destination(user_query)

    # 2. Only fall back to current_location if the query has no explicit destination
    if not destination:
        loc = state.get("current_location", "")
        if loc and not loc.strip().lower().endswith(" unavailable"):
            destination = loc

    # 3. Get user interests for query enrichment
    interests = _get_user_interests(state)

    # 4. Build enriched query and retrieve from KB
    enriched_query = _build_enriched_query(user_query, destination, interests)

    logger.debug(
        "[RecommendationsRAG] query=%r destination=%r interests=%r enriched=%r",
        user_query, destination, interests, enriched_query,
    )

    try:
        result = answer_tourist_query(query=enriched_query, use_llm=True)
    except Exception as exc:
        logger.error("[RecommendationsRAG] answer_tourist_query failed: %s", exc)
        # Safe fallback
        no_data_msg = (
            f"I don't have detailed attraction or restaurant listings "
            f"{'for ' + destination if destination else 'in my knowledge base'} "
            "at the moment. You can call the Egyptian Tourism Authority hotline at **19654** "
            "for up-to-date recommendations."
        )
        return {
            "retrieved_context": "",
            "candidate_response": no_data_msg,
            "response": no_data_msg,
            "retrieval_sources": [],
            "rag_status": "error",
            "out_of_scope": False,
            "help_domain": "recommendations",
            "current_intent": "recommendations",
            "next_agent": "recommendations_completed",
        }

    # Handle out-of-scope (e.g. "restaurants in Paris")
    if result.status == "out_of_scope":
        logger.info("[RecommendationsRAG] out_of_scope for query=%r", user_query)
        return {
            "retrieved_context": "",
            "candidate_response": result.answer,
            "response": result.answer,
            "retrieval_sources": result.sources,
            "rag_status": "out_of_scope",
            "out_of_scope": True,
            "verified_response": result.answer,
            "is_verified": True,
            "help_domain": "recommendations",
            "current_intent": "recommendations",
            "next_agent": "recommendations_completed",
        }

    # Handle no-answer gracefully — do NOT return hardcoded Cairo placeholders
    if result.status == "no_answer":
        dest_label = f" for {destination}" if destination else " in Egypt"
        no_data_msg = (
            f"I don't have detailed place recommendations{dest_label} "
            "in my current knowledge base. For up-to-date recommendations, "
            "you can call the Egyptian Tourism Authority hotline at **19654** "
            "or check the official Egypt tourism website."
        )
        logger.info("[RecommendationsRAG] no_answer for destination=%r", destination)
        return {
            "retrieved_context": result.context,
            "candidate_response": no_data_msg,
            "response": no_data_msg,
            "retrieval_sources": result.sources,
            "rag_status": "no_answer",
            "out_of_scope": False,
            "help_domain": "recommendations",
            "current_intent": "recommendations",
            "next_agent": "recommendations_completed",
        }

    # Successful retrieval
    logger.debug("[RecommendationsRAG] answered. context_len=%d", len(result.context))
    return {
        "retrieved_context": result.context,
        "candidate_response": result.answer,
        "response": result.answer,
        "retrieval_sources": result.sources,
        "rag_status": "answered",
        "out_of_scope": False,
        "help_domain": "recommendations",
        "current_intent": "recommendations",
        "next_agent": "recommendations_completed",
    }


# Backward-compatible alias
recommendations_agent_node = recommendations_node
