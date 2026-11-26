"""
Tourist Help Subgraph - Intent & Location Resolution Node (Story 2.1)
Module: src/agents/tourist_help/intent_resolver.py
"""

import re
from typing import Dict, Any, Optional, Tuple
import logging
from src.services.location_services import reverse_geocode

logger = logging.getLogger(__name__)

# =========================================================================
# Multilingual Keyword Dictionaries for Intent Resolution
# Fix (Problem 9): Use whole-word boundary matching (\b) everywhere.
# A keyword like "eat" must NOT match "great", "heat", "weather", etc.
# =========================================================================

EMERGENCY_KEYWORDS = [
    "emergency", "police", "ambulance", "hospital", "doctor", "accident",
    "robbed", "stolen", "thief", "harass", "harassment", "fire", "danger",
    "help me", "tourist police", "scam", "urgent", "126", "122", "123", "180",
    "طوارئ", "شرطة", "إسعاف", "مستشفى", "حادث", "سرقة", "مطافئ", "نجدة",
]

# Extended advisory keywords to cover transport info, tipping, safety advisories,
# hospitals, and common tourist questions that must NOT go to the Planner.
ADVISORY_KEYWORDS = [
    # Visa & entry
    "visa", "entry", "passport", "transit", "advisory", "advisories", "customs",
    "embassy", "immigration", "evisa", "e-visa",
    # Rules & regulations
    "rules", "regulations", "laws", "dress code", "sim card", "currency",
    # Transport info (factual, not booking)
    "transport", "metro", "bus", "train", "taxi", "taxis", "cab", "uber",
    "careem", "ferry", "felucca",
    # Tipping & money
    "baksheesh", "tipping", "tip",
    # Safety & advisories
    "safety", "safe", "dangerous", "danger", "unsafe", "risk", "crime",
    "scams", "pickpocket", "harassment",
    # Medical (non-emergency info)
    "pharmacy", "clinic",
    # Arabic equivalents
    "فيزا", "تأشيرة", "جواز", "مواصلات", "قواعد", "أمان", "أمن",
]

RECOMMENDATION_KEYWORDS = [
    "recommend", "recommendation", "restaurant", "food", "koshary", "cafe",
    "cafes", "places", "sights", "attractions", "best spots", "what to see",
    "museum", "pyramids", "near me", "nearby",
]

PLANNER_KEYWORDS = [
    "itinerary", "plan", "days in", "day trip", "schedule",
]


# =========================================================================
# Location & Coordinate Extraction (Story 2.1)
# =========================================================================

def extract_coordinates(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Extracts latitude and longitude from multiple possible representations in state.
    Supports:
      - state["raw_coordinates"] = {"lat": 30.0444, "lng": 31.2357} or {"lat": 30.0444, "lon": 31.2357}
      - state["coordinates"] = ...
      - state["current_location"] if provided as a dict
      - state["trip_requirements"].current_location (dict or Pydantic model)
      - tuple / list of [lat, lon]
    """
    candidate = None

    if "raw_coordinates" in state and state["raw_coordinates"]:
        candidate = state["raw_coordinates"]
    elif "coordinates" in state and state["coordinates"]:
        candidate = state["coordinates"]
    elif "current_location" in state and isinstance(state["current_location"], dict):
        candidate = state["current_location"]
    elif "trip_requirements" in state and state["trip_requirements"]:
        reqs = state["trip_requirements"]
        if hasattr(reqs, "current_location"):
            candidate = reqs.current_location
        elif isinstance(reqs, dict):
            candidate = reqs.get("current_location")

    if candidate is None:
        return None

    # Handle (lat, lon) tuple or list
    if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
        try:
            lat = float(candidate[0])
            lon = float(candidate[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except (ValueError, TypeError):
            return None

    # Handle dictionary representation
    if isinstance(candidate, dict):
        lat = candidate.get("lat") or candidate.get("latitude")
        lon = candidate.get("lon") or candidate.get("lng") or candidate.get("longitude")
        if lat is not None and lon is not None:
            try:
                lat = float(lat)
                lon = float(lon)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return lat, lon
            except (ValueError, TypeError):
                return None

    return None


def resolve_location_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for Story 2.1 Coordinate Resolution.
    Converts incoming raw coordinates into a human-readable location address
    before any downstream agent (Help Agent, Planner Agent, etc.) inspects the state.

    Overwrites raw coordinates with the formatted `current_location` string.
    """
    coords = extract_coordinates(state)

    if coords is None:
        return {}

    lat, lon = coords
    location_response = reverse_geocode(lat=lat, lon=lon)

    if location_response.error:
        logger.warning(
            f"[LocationResolver] Geocoding error for ({lat}, {lon}): {location_response.error}"
        )

    return {
        "current_location": location_response.formatted_address,
        "raw_coordinates": None,
        "location_details": {
            "city": location_response.city,
            "country": location_response.country,
            "formatted_address": location_response.formatted_address,
            "error": location_response.error,
        },
    }

# =========================================================================
# Intent Classification & Parser Nodes (Story 2.1)
# =========================================================================

def _kw_match(keyword: str, text: str) -> bool:
    """
    Whole-word boundary match for a keyword in text.
    Multi-word keywords (e.g. "help me") use a simple substring check
    since \b doesn't work reliably on space-separated phrases.
    """
    if " " in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def parse_tourist_intent(user_text: str) -> str:
    """
    Classifies a user query into one of the specialized Tourist Help categories:
    - 'emergency': Distress, police, medical, fire, tourist police, accidents.
    - 'advisories': Visa, passport, customs, transit rules, embassy, dress code,
                    tipping, transport info, safety, hospital info.
    - 'recommendations': Local dining, cafes, tourist attractions, nearby activities.
    - 'plan_trip': Full itinerary planning (multi-day schedules).
    - 'fallback': Ambiguous or unclassified requests.

    Fix (Problem 9): Uses whole-word regex matching (\b) so that partial
    substring overlaps like "eat" in "great" or "bus" in "business" do not
    trigger false-positive intents.
    """
    text = user_text.lower().strip()

    # 1. Emergency Assistance (Story 2.2) — highest priority
    if any(_kw_match(k, text) for k in EMERGENCY_KEYWORDS):
        return "emergency"

    # 2. General Advisories, Transit/Visa Rules, Tipping, Safety (Story 2.3)
    if any(_kw_match(k, text) for k in ADVISORY_KEYWORDS):
        return "advisories"

    # 3. Contextual Place Recommendations (Story 2.4)
    if any(_kw_match(k, text) for k in RECOMMENDATION_KEYWORDS):
        return "recommendations"

    # 4. Multi-day Planning Request (Delegation to Planner)
    if any(_kw_match(k, text) for k in PLANNER_KEYWORDS):
        return "plan_trip"

    return "fallback"


# Alias for backward compatibility with PR #3
resolve_tourist_help_intent = parse_tourist_intent


def intent_parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: extracts the user query, classifies the sub-intent,
    and updates both `current_intent` and `help_domain` in state.

    Fix (Problem 10): Always re-parses from the latest HumanMessage.
    The previous early-return guard ("if current_intent already set, skip")
    caused stale intent inheritance across independent turns in the same session.
    """
    # Always extract the latest human message — never re-use a stale current_intent.
    user_prompt = None

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_prompt = str(msg.content).strip()
            break
        if isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]).lower() in {"human", "user"}:
            user_prompt = str(msg[1]).strip()
            break

    # Only fall back to state["user_query"] if the message list is empty
    if not user_prompt:
        user_prompt = state.get("user_query", "Hello")

    parsed_intent = parse_tourist_intent(user_prompt)

    logger.debug("[IntentParser] query=%r -> intent=%r", user_prompt, parsed_intent)

    return {
        "user_query": user_prompt,
        "current_intent": parsed_intent,
        "help_domain": parsed_intent,
        "next_help_node": parsed_intent,
        "intent": f"Tourist help parsed intent: {parsed_intent}",
    }


# Alias for backward compatibility with PR #3
intent_resolver_node = intent_parser_node


# =========================================================================
# Conditional Routing Logic (Story 2.1 Item 3)
# =========================================================================

def route_request(state: Dict[str, Any]) -> str:
    """
    Conditional edge function:
    Evaluates `current_intent` or `help_domain` and returns the target node name:
    - 'emergency' -> 'emergency' (or 'emergency_rag')
    - 'advisories' -> 'advisories' (or 'advisories_rag')
    - 'recommendations' -> 'recommendations'
    - 'plan_trip' -> 'planner_agent'
    - 'fallback' -> 'fallback'
    """
    raw_intent = str(state.get("current_intent") or state.get("help_domain") or "").lower().strip()

    if raw_intent in ["emergency", "emergency_rag", "help_agent", "police", "medical"]:
        return "emergency"
    elif raw_intent in ["advisories", "advisories_rag", "tourist_rag", "rules", "visa", "transit"]:
        return "advisories"
    elif raw_intent in ["recommendations", "places", "places_recommendation"]:
        return "recommendations"
    elif raw_intent in ["plan_trip", "planner", "planner_agent"]:
        return "planner_agent"

    return "fallback"


# Alias for backward compatibility with PR #3
route_help_subintent = route_request


def fallback_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback agent node within Tourist Help Subgraph for ambiguous queries.
    Attempts a KB search before returning the generic fallback message.
    """
    # Try the KB for queries that failed keyword classification but might be answerable
    user_query = state.get("user_query") or ""
    if user_query:
        try:
            from src.rag.tourist_rag.answer import answer_tourist_query
            result = answer_tourist_query(query=user_query, use_llm=True)
            if result.status == "answered":
                return {
                    "retrieved_context": result.context,
                    "candidate_response": result.answer,
                    "response": result.answer,
                    "retrieval_sources": result.sources,
                    "rag_status": "answered",
                    "out_of_scope": False,
                    "help_domain": "general",
                    "current_intent": "fallback",
                    "next_agent": "fallback_completed",
                }
            if result.status == "out_of_scope":
                return {
                    "retrieved_context": "",
                    "candidate_response": result.answer,
                    "response": result.answer,
                    "retrieval_sources": result.sources,
                    "rag_status": "out_of_scope",
                    "out_of_scope": True,
                    "verified_response": result.answer,
                    "is_verified": True,
                    "help_domain": "general",
                    "current_intent": "fallback",
                    "next_agent": "fallback_completed",
                }
        except Exception as exc:
            logger.debug("[FallbackAgent] KB search failed: %s", exc)

    fallback_response = (
        "I'm here to help with your trip in Egypt! "
        "You can ask for emergency assistance, visa/travel advisories, or nearby place recommendations."
    )
    return {
        "response": fallback_response,
        "verified_response": fallback_response,
        "candidate_response": fallback_response,
        "retrieved_context": "",
        "retrieval_sources": [],
        "rag_status": "no_answer",
        "out_of_scope": False,
        "help_domain": "general",
        "current_intent": "fallback",
        "next_agent": "fallback_completed",
    }
