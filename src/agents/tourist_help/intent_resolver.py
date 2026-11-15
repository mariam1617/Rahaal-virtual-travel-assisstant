"""
Tourist Help Subgraph - Intent & Location Resolution Node (Story 2.1)
Module: src/agents/tourist_help/intent_resolver.py
"""

from typing import Dict, Any, Optional, Tuple
import logging
from src.services.location_services import reverse_geocode

logger = logging.getLogger(__name__)


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
    LangGraph node function for Story 2.1 Coordinate Resolution.
    Converts incoming raw coordinates into a human-readable location address
    before any downstream agent (Help Agent, Planner Agent, etc.) inspects the state.
    
    Overwrites raw coordinates with the formatted `current_location` string.
    """
    coords = extract_coordinates(state)
    
    if coords is None:
        # If no raw coordinates are provided, preserve existing state untouched
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
        "raw_coordinates": None,  # Overwrite raw coordinates so downstream agents use human-readable address
        "location_details": {
            "city": location_response.city,
            "country": location_response.country,
            "formatted_address": location_response.formatted_address,
            "error": location_response.error,
        }
    }


def parse_tourist_intent(user_text: str) -> str:
    """
    Classifies a user query into one of the specialized Tourist Help categories:
    - 'emergency': Distress, police, medical, fire, tourist police, accidents.
    - 'advisories': Visa, passport, customs, transit rules, embassy, dress code, tipping.
    - 'recommendations': Local dining, cafes, tourist attractions, nearby activities.
    - 'plan_trip': Full itinerary planning.
    - 'fallback': Ambiguous or unclassified requests.
    """
    text = user_text.lower().strip()

    # 1. Emergency Assistance (Story 2.2)
    emergency_keywords = [
        "emergency", "police", "ambulance", "hospital", "stolen", "theft",
        "robbed", "fire", "accident", "urgent", "126", "122", "123", "180",
        "tourist police", "harassment", "scam", "danger", "help me"
    ]
    if any(k in text for k in emergency_keywords):
        return "emergency"

    # 2. General Advisories & Transit/Visa Rules (Story 2.3)
    advisories_keywords = [
        "visa", "passport", "entry requirement", "customs", "advisory", "advisories",
        "embassy", "transit", "metro", "bus", "transport", "rules", "dress code",
        "tipping", "baksheesh", "sim card", "laws", "regulations"
    ]
    if any(k in text for k in advisories_keywords):
        return "advisories"

    # 3. Contextual Place Recommendations (Story 2.4)
    recommendation_keywords = [
        "recommend", "restaurant", "eat", "food", "koshary", "cafe", "places",
        "sights", "attractions", "visit", "near me", "nearby", "best spots",
        "what to see", "museum", "pyramids"
    ]
    if any(k in text for k in recommendation_keywords):
        return "recommendations"

    # 4. Multi-day Planning Request (Delegation to Planner)
    planner_keywords = ["itinerary", "plan", "days in", "day trip", "schedule"]
    if any(k in text for k in planner_keywords):
        return "plan_trip"

    return "fallback"


def intent_parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for Story 2.1 Item 3:
    Extracts the user inquiry, classifies it, and updates `current_intent`
    in the state dictionary before branching.
    """
    # If already set to a valid non-default intent, preserve it
    current_intent = state.get("current_intent")
    if current_intent and current_intent not in ["unclassified", "unknown", "None"]:
        return state

    messages = state.get("messages", [])
    if not messages:
        user_prompt = "Hello"
    else:
        last_msg = messages[-1]
        if isinstance(last_msg, tuple):
            user_prompt = last_msg[1]
        elif hasattr(last_msg, "content"):
            user_prompt = last_msg.content
        else:
            user_prompt = str(last_msg)

    parsed_intent = parse_tourist_intent(user_prompt)

    return {
        **state,
        "current_intent": parsed_intent,
        "intent": f"Tourist help parsed intent: {parsed_intent}"
    }


def route_request(state: Dict[str, Any]) -> str:
    """
    Conditional edge function for Story 2.1 Item 3:
    Evaluates `current_intent` in the state and returns the exact target node name:
    - 'emergency' (or 'help_agent') -> 'emergency'
    - 'advisories' -> 'advisories'
    - 'recommendations' -> 'recommendations'
    - 'plan_trip' (or 'planner_agent') -> 'planner_agent'
    - 'fallback' -> 'fallback'
    """
    raw_intent = str(state.get("current_intent", "")).lower().strip()

    if raw_intent in ["emergency", "help_agent", "police", "medical"]:
        return "emergency"
    elif raw_intent in ["advisories", "tourist_rag", "rules", "visa", "transit"]:
        return "advisories"
    elif raw_intent in ["recommendations", "places", "places_recommendation"]:
        return "recommendations"
    elif raw_intent in ["plan_trip", "planner", "planner_agent"]:
        return "planner_agent"
    
    return "fallback"


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
        "next_agent": "fallback_completed"
    }
