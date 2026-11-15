"""
Unit & Integration Tests for Planner Subgraph (Story 3.1, 3.2, 5.1)
Covers Base Tools, Active Profiling, Draft Generation, Constraint Validation,
Constraint Repair Loop, and Cyclic LangGraph Execution.
"""

import pytest
from src.tools.maps import (
    search_local_places,
    calculate_travel_time,
    EGYPT_PLACES_DATABASE,
)
from src.tools.weather import get_weather_forecast


# =========================================================================
# 1. Milestone 1: Base Tools Tests (Story 3.1)
# =========================================================================

def test_search_local_places_cairo():
    """Verify places lookup retrieves Cairo attractions with cost and duration."""
    places = search_local_places(destination="Cairo", limit=10)
    assert len(places) > 0
    names = [p["name"] for p in places]
    assert any("Pyramids" in n for n in names)
    assert any("Museum" in n for n in names)
    
    first = places[0]
    assert "estimated_cost_usd" in first
    assert "recommended_duration_hours" in first
    assert "coordinates" in first


def test_search_local_places_category_filter():
    """Verify category filtering for dining and museums."""
    dining_places = search_local_places(destination="Cairo", category="dining")
    assert len(dining_places) > 0
    assert all(p["category"] == "dining" for p in dining_places)
    assert any("Abou Tarek" in p["name"] for p in dining_places)


def test_search_local_places_query_search():
    """Verify specific query search."""
    gem = search_local_places(query="Grand Egyptian Museum", destination="Cairo")
    assert len(gem) > 0
    assert "GEM" in gem[0]["id"] or "Grand Egyptian Museum" in gem[0]["name"]


def test_calculate_travel_time_known_routes():
    """Verify travel time and distance calculation between known landmarks."""
    route = calculate_travel_time("Downtown Cairo", "Giza Pyramids")
    assert route["duration_minutes"] >= 30
    assert route["distance_km"] > 10
    assert route["estimated_transit_cost_usd"] > 0


def test_calculate_travel_time_fallback():
    """Verify realistic fallback for unknown intra-city route."""
    route = calculate_travel_time("Zamalek Hotel", "Cairo Tower")
    assert route["duration_minutes"] > 0
    assert route["distance_km"] > 0


def test_get_weather_forecast_luxor_heat_advisory():
    """Verify Luxor weather flags midday heat advisory."""
    forecast = get_weather_forecast("Luxor")
    assert forecast["destination"] == "Luxor"
    assert forecast["temperature_high_c"] >= 30
    assert forecast["heat_advisory"] is True
    assert "morning" in forecast["advice"].lower()


def test_get_weather_forecast_alexandria():
    """Verify Alexandria coastal weather forecast."""
    forecast = get_weather_forecast("Alexandria")
    assert forecast["destination"] == "Alexandria"
    assert forecast["temperature_high_c"] < 30
    assert forecast["rain_probability_pct"] >= 0
