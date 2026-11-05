"""
Tourist Help Intent & Location Resolver Node (Story 2.1 / Dev 1 Stub)
Branches user queries between emergency assistance, general advisories/visas, and recommendations.
"""

from typing import Dict, Any


EMERGENCY_KEYWORDS = [
    "emergency", "police", "ambulance", "hospital", "doctor", "accident",
    "robbed", "stolen", "thief", "harass", "fire", "danger", "help me",
    "طوارئ", "شرطة", "إسعاف", "مستشفى", "حادث", "سرقة", "مطافئ"
]

ADVISORY_KEYWORDS = [
    "visa", "entry", "passport", "transit", "advisory", "customs", "rules",
    "baksheesh", "tipping", "scam", "metro", "bus", "transport", "train",
    "فيزا", "تأشيرة", "جواز", "مواصلات", "قواعد"
]


def resolve_tourist_help_intent(query: str) -> str:
    """Heuristic classifier to resolve sub-domain within Tourist Help."""
    lower_q = query.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in lower_q:
            return "emergency"
    for kw in ADVISORY_KEYWORDS:
        if kw in lower_q:
            return "advisories"
    return "recommendations"


def intent_resolver_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: parses user query and GPS coordinates, determining the target Help sub-agent.
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        user_query = getattr(last_msg, "content", None) or (last_msg[1] if isinstance(last_msg, tuple) else str(last_msg))
    else:
        user_query = state.get("user_query", "")

    sub_intent = resolve_tourist_help_intent(user_query)

    return {
        **state,
        "user_query": user_query,
        "help_domain": sub_intent,
        "next_help_node": sub_intent
    }
