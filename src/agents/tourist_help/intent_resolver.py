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
