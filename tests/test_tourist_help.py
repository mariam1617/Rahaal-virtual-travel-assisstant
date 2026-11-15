"""
Unit & Integration Tests for Tourist Help Subgraph - Coordinate Resolution (Story 2.1)
"""

from unittest.mock import patch
from src.agents.tourist_help.intent_resolver import (
    extract_coordinates,
    resolve_location_node,
)
from src.services.location import LocationResponse
from src.state.schema import TripRequirements
from src.agents.supervisor.router import create_supervisor_graph


# ---------------------------------------------------------
# 1. Coordinate Extraction Tests
# ---------------------------------------------------------
def test_extract_coordinates_from_raw_coordinates_dict_lng():
    state = {"raw_coordinates": {"lat": 30.0444, "lng": 31.2357}}
    coords = extract_coordinates(state)
    assert coords == (30.0444, 31.2357)


def test_extract_coordinates_from_coordinates_dict_lon():
    state = {"coordinates": {"lat": 31.2001, "lon": 29.9187}}
    coords = extract_coordinates(state)
    assert coords == (31.2001, 29.9187)


def test_extract_coordinates_from_tuple():
    state = {"raw_coordinates": (25.6872, 32.6396)}
    coords = extract_coordinates(state)
    assert coords == (25.6872, 32.6396)


def test_extract_coordinates_from_trip_requirements_pydantic():
    trip = TripRequirements(
        destination="Cairo",
        current_location={"lat": 30.0444, "lng": 31.2357}
    )
    state = {"trip_requirements": trip}
    coords = extract_coordinates(state)
    assert coords == (30.0444, 31.2357)


def test_extract_coordinates_invalid_bounds():
    state = {"raw_coordinates": {"lat": 120.0, "lon": 31.2357}}  # Lat > 90
    assert extract_coordinates(state) is None


def test_extract_coordinates_missing():
    state = {"messages": [("user", "Hello")]}
    assert extract_coordinates(state) is None


# ---------------------------------------------------------
# 2. Coordinate Resolution Node Tests
# ---------------------------------------------------------
@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_resolve_location_node_success(mock_reverse_geocode):
    mock_reverse_geocode.return_value = LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Tahrir Square, Cairo, Egypt",
        error=None
    )

    state = {
        "raw_coordinates": {"lat": 30.0444, "lng": 31.2357},
        "messages": [("user", "What can I do nearby?")]
    }

    result = resolve_location_node(state)

    mock_reverse_geocode.assert_called_once_with(lat=30.0444, lon=31.2357)
    assert result["current_location"] == "Tahrir Square, Cairo, Egypt"
    assert result["raw_coordinates"] is None
    assert result["location_details"]["city"] == "Cairo"
    assert result["location_details"]["country"] == "Egypt"
    assert result["location_details"]["error"] is None
    # Original state preserved
    assert result["messages"] == [("user", "What can I do nearby?")]


def test_resolve_location_node_no_coordinates_passthrough():
    state = {
        "messages": [("user", "Hello")],
        "current_location": "Pre-existing Cairo"
    }

    result = resolve_location_node(state)
    # When no coordinates exist, state is returned untouched
    assert result == state
    assert result["current_location"] == "Pre-existing Cairo"


@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_resolve_location_node_api_error_fallback(mock_reverse_geocode):
    mock_reverse_geocode.return_value = LocationResponse(
        city=None,
        country=None,
        formatted_address="Location unavailable",
        error="Connection timed out"
    )

    state = {
        "raw_coordinates": {"lat": 30.0444, "lng": 31.2357},
        "messages": [("user", "Help")]
    }
    result = resolve_location_node(state)

    assert result["current_location"] == "Location unavailable"
    assert result["raw_coordinates"] is None
    assert result["location_details"]["error"] == "Connection timed out"
    assert result["messages"] == [("user", "Help")]


# ---------------------------------------------------------
# 3. End-to-End Supervisor Graph Integration Tests
# ---------------------------------------------------------
@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_supervisor_graph_resolves_location_and_routes(mock_reverse_geocode):
    mock_reverse_geocode.return_value = LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Zamalek, Cairo, Egypt",
        error=None
    )

    graph = create_supervisor_graph()
    initial_state = {
        "messages": [("user", "Where can I eat authentic Koshary near me right now?")],
        "raw_coordinates": {"lat": 30.0609, "lng": 31.2197}
    }

    final_state = graph.invoke(initial_state)

    # 1. Location should be resolved and raw coordinates cleared
    assert final_state["current_location"] == "Zamalek, Cairo, Egypt"
    assert final_state["raw_coordinates"] is None

    # 2. Supervisor should route to tourist_help with high confidence
    assert final_state["next_agent"] == "tourist_help"
    assert final_state["is_egypt_scoped"] is True


# ---------------------------------------------------------
# 4. FastAPI Endpoint Integration Test
# ---------------------------------------------------------
@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_api_supervisor_chat_with_coordinates(mock_reverse_geocode):
    from fastapi.testclient import TestClient
    from src.main import app

    mock_reverse_geocode.return_value = LocationResponse(
        city="Giza",
        country="Egypt",
        formatted_address="Pyramids of Giza, Giza, Egypt",
        error=None
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/supervisor/chat",
        json={
            "user_id": "test_user",
            "session_id": "test_sess",
            "message": "Where can I find a good restaurant nearby?",
            "coordinates": {"lat": 29.9792, "lng": 31.1342}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["routed_to"] == "tourist_help"
    assert data["current_location"] == "Pyramids of Giza, Giza, Egypt"


# ---------------------------------------------------------
# 5. Intent Parsing & Subgraph Routing Tests (Story 2.1 Item 3)
# ---------------------------------------------------------
import pytest
from src.agents.tourist_help.intent_resolver import (
    parse_tourist_intent,
    intent_parser_node,
    route_request,
)
from src.agents.tourist_help.graph import create_tourist_help_graph


@pytest.mark.parametrize("query,expected_intent", [
    ("I was robbed, call the police immediately!", "emergency"),
    ("What is the phone number for the ambulance in Cairo?", "emergency"),
    ("Tourist police contact number in Hurghada", "emergency"),
    ("Do British citizens need an entry visa on arrival?", "advisories"),
    ("What are the rules and customs for the Cairo metro?", "advisories"),
    ("What is the dress code when visiting mosques?", "advisories"),
    ("Where can I eat authentic Koshary nearby?", "recommendations"),
    ("Recommend top historical sights and cafes in Zamalek", "recommendations"),
    ("Plan a 3-day itinerary for Luxor and Aswan", "plan_trip"),
    ("Hello there!", "fallback"),
    ("What is the meaning of life?", "fallback"),
])
def test_parse_tourist_intent_benchmark(query: str, expected_intent: str):
    assert parse_tourist_intent(query) == expected_intent


def test_intent_parser_node():
    state = {"messages": [("user", "Someone stole my wallet, call tourist police!")]}
    updated = intent_parser_node(state)
    assert updated["current_intent"] == "emergency"
    assert "emergency" in updated["intent"]


@pytest.mark.parametrize("intent,expected_branch", [
    ("emergency", "emergency"),
    ("help_agent", "emergency"),
    ("police", "emergency"),
    ("advisories", "advisories"),
    ("tourist_rag", "advisories"),
    ("visa", "advisories"),
    ("recommendations", "recommendations"),
    ("places", "recommendations"),
    ("plan_trip", "planner_agent"),
    ("planner_agent", "planner_agent"),
    ("unknown", "fallback"),
    ("", "fallback"),
])
def test_route_request_mapping(intent: str, expected_branch: str):
    state = {"current_intent": intent}
    assert route_request(state) == expected_branch


def test_tourist_help_graph_emergency_branch():
    graph = create_tourist_help_graph()
    initial_state = {
        "messages": [("user", "Emergency! I need the tourist police right now.")]
    }
    result = graph.invoke(initial_state)

    assert result["current_intent"] == "emergency"
    assert result["next_agent"] == "emergency_completed"
    assert "126" in result["response"]


def test_tourist_help_graph_advisories_branch():
    graph = create_tourist_help_graph()
    initial_state = {
        "messages": [("user", "Do US citizens need a visa to enter Egypt?")]
    }
    result = graph.invoke(initial_state)

    assert result["current_intent"] == "advisories"
    assert result["next_agent"] == "advisories_completed"
    assert "Visa on Arrival" in result["response"]


@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_tourist_help_graph_recommendations_branch(mock_reverse_geocode):
    mock_reverse_geocode.return_value = LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Downtown Cairo, Egypt",
        error=None
    )

    graph = create_tourist_help_graph()
    initial_state = {
        "messages": [("user", "Can you recommend good places to eat nearby?")],
        "raw_coordinates": {"lat": 30.0444, "lng": 31.2357}
    }
    result = graph.invoke(initial_state)

    assert result["current_intent"] == "recommendations"
    assert result["current_location"] == "Downtown Cairo, Egypt"
    assert result["raw_coordinates"] is None
    assert result["next_agent"] == "recommendations_completed"
    assert "Downtown Cairo" in result["response"]


def test_tourist_help_graph_planner_delegation_branch():
    graph = create_tourist_help_graph()
    initial_state = {
        "messages": [("user", "Plan a 4-day itinerary for Cairo and Alexandria")]
    }
    result = graph.invoke(initial_state)

    assert result["current_intent"] == "plan_trip"
    assert result["next_agent"] == "planner"


def test_tourist_help_graph_fallback_branch():
    graph = create_tourist_help_graph()
    initial_state = {
        "messages": [("user", "Hello there")]
    }
    result = graph.invoke(initial_state)

    assert result["current_intent"] == "fallback"
    assert result["next_agent"] == "fallback_completed"
    assert "help with your trip in Egypt" in result["response"]


