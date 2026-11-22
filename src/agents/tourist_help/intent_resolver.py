"""
Tourist Help Subgraph - Intent & Location Resolution Node (Story 2.1)
Module: src/agents/tourist_help/intent_resolver.py
"""

from typing import Dict, Any, Optional, Tuple
import logging
from src.services.location_services import reverse_geocode

logger = logging.getLogger(__name__)

# =========================================================================
# Multilingual Keyword Dictionaries for Intent Resolution
# =========================================================================

EMERGENCY_KEYWORDS = [
    "emergency", "police", "ambulance", "hospital", "doctor", "accident",
    "robbed", "stolen", "thief", "harass", "harassment", "fire", "danger",
    "help me", "tourist police", "scam", "urgent", "126", "122", "123", "180",
    "طوارئ", "شرطة", "إسعاف", "مستشفى", "حادث", "سرقة", "مطافئ", "نجدة"
]

ADVISORY_KEYWORDS = [
    "visa", "entry", "passport", "transit", "advisory", "advisories", "customs",
    "rules", "baksheesh", "tipping", "metro", "bus", "transport", "train",
    "dress code", "sim card", "laws", "regulations", "embassy", "currency",
    "فيزا", "تأشيرة", "جواز", "مواصلات", "قواعد"
]

RECOMMENDATION_KEYWORDS = [
    "recommend", "restaurant", "eat", "food", "koshary", "cafe", "places",
    "sights", "attractions", "visit", "near me", "nearby", "best spots",
    "what to see", "museum", "pyramids"
]

PLANNER_KEYWORDS = [
    "itinerary", "plan", "days in", "day trip", "schedule"
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
    # TODO (Problem 5): A geocoding failure leaks into user answers.
    # reverse_geocode returns formatted_address="Location unavailable" on error, and this node writes that
    # into current_location. Downstream emergency and recommendation nodes then print
    # "Emergency Assistance for Location unavailable".
    # Handle error cases gracefully (e.g. set current_location to None or fallback to "Egypt"),
    # and provide structured fields: governorate (state/province), neighborhood, country code, lat/lon.
    coords = extract_coordinates(state)
    
    if coords is None:
        return state

    lat, lon = coords
    location_response = reverse_geocode(lat=lat, lon=lon)

    if location_response.error:
        logger.warning(
            f"[LocationResolver] Geocoding error for ({lat}, {lon}): {location_response.error}"
        )

    return {
        **state,
        "current_location": location_response.formatted_address,
        "raw_coordinates": None,
        "location_details": {
            "city": location_response.city,
            "country": location_response.country,
            "formatted_address": location_response.formatted_address,
            "error": location_response.error,
        }
    }


# =========================================================================
# Intent Classification & Parser Nodes (Story 2.1)
# =========================================================================

def parse_tourist_intent(user_text: str) -> str:
    """
    Classifies a user query into one of the specialized Tourist Help categories:
    - 'emergency': Distress, police, medical, fire, tourist police, accidents.
    - 'advisories': Visa, passport, customs, transit rules, embassy, dress code, tipping.
    - 'recommendations': Local dining, cafes, tourist attractions, nearby activities.
    - 'plan_trip': Full itinerary planning.
    - 'fallback': Ambiguous or unclassified requests.
    """
    # TODO (Problem 9): Substring keyword matching misfires!
    # Using `any(k in text ...)` means:
    # - "eat" matches "great" (redirecting to recommendations)
    # - "bus" matches "business" (redirecting to advisories)
    # - "scam" in "common scams to avoid" incorrectly sends advice queries to emergency.
    # Replace substring matching with regex whole-word boundaries (rf"\b{re.escape(k)}\b") or intent classification.
    text = user_text.lower().strip()

    # 1. Emergency Assistance (Story 2.2)
    if any(k in text for k in EMERGENCY_KEYWORDS):
        return "emergency"

    # 2. General Advisories & Transit/Visa Rules (Story 2.3)
    if any(k in text for k in ADVISORY_KEYWORDS):
        return "advisories"

    # 3. Contextual Place Recommendations (Story 2.4)
    if any(k in text for k in RECOMMENDATION_KEYWORDS):
        return "recommendations"

    # 4. Multi-day Planning Request (Delegation to Planner)
    if any(k in text for k in PLANNER_KEYWORDS):
        return "plan_trip"

    return "fallback"


# Alias for backward compatibility with PR #3
resolve_tourist_help_intent = parse_tourist_intent


def intent_parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: extracts the user query, classifies the sub-intent,
    and updates both `current_intent` and `help_domain` in state.
    """
    # TODO (Problem 10): Stale intent bug.
    # intent_parser_node returns early if current_intent is already in state, and reads user_query first.
    # On a multi-turn conversation or second turn, the old intent or stale query wins, ignoring the latest user message.
    # Always extract the latest user message from state["messages"][-1] rather than reusing stale current_intent/user_query.
    current_intent = state.get("current_intent") or state.get("help_domain")
    if current_intent and current_intent not in ["unclassified", "unknown", "None", "general"]:
        return {
            **state,
            "help_domain": current_intent,
            "current_intent": current_intent,
            "next_help_node": current_intent
        }

    # Extract user prompt from messages or user_query
    user_prompt = state.get("user_query")
    if not user_prompt:
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, tuple):
                user_prompt = last_msg[1]
            elif hasattr(last_msg, "content"):
                user_prompt = last_msg.content
            else:
                user_prompt = str(last_msg)
        else:
            user_prompt = "Hello"

    parsed_intent = parse_tourist_intent(user_prompt)

    return {
        **state,
        "user_query": user_prompt,
        "current_intent": parsed_intent,
        "help_domain": parsed_intent,
        "next_help_node": parsed_intent,
        "intent": f"Tourist help parsed intent: {parsed_intent}"
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
    """
    fallback_response = (
        "I'm here to help with your trip in Egypt! "
        "You can ask for emergency assistance, visa/travel advisories, or nearby place recommendations."
    )
    return {
        **state,
        "response": fallback_response,
        "verified_response": fallback_response,
        "candidate_response": fallback_response,
        "help_domain": "general",
        "current_intent": "fallback",
        "next_agent": "fallback_completed"
    }
