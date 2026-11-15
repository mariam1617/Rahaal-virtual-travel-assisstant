"""
Unit & Integration Tests for Tourist Help Subgraph (Epic 2)
Covers Coordinate Resolution, Intent Resolution, RAG Candidate Generation,
Anti-Hallucination Verification Guardrails, Subgraph Execution, and API Endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.tourist_help.schemas import (
    RAGInputContext,
    VerificationResult,
)
from src.agents.tourist_help.prompts import (
    OFFICIAL_EGYPT_EMERGENCY_NUMBERS,
    DEFAULT_EMERGENCY_FALLBACK,
    DEFAULT_ADVISORY_FALLBACK,
    DEFAULT_GENERAL_FALLBACK,
    HELP_VERIFICATION_SYSTEM_PROMPT,
)
from src.agents.tourist_help.verification import (
    detect_foreign_emergency_numbers,
    extract_phone_numbers,
    heuristic_verification_fallback,
    verify_rag_response,
    help_verification_node,
)
from src.agents.tourist_help.intent_resolver import (
    extract_coordinates,
    resolve_location_node,
    parse_tourist_intent,
    intent_parser_node,
    route_request,
)
from src.agents.tourist_help.graph import create_tourist_help_graph, tourist_help_app
from src.services.location import LocationResponse
from src.state.schema import TripRequirements
from src.agents.supervisor.router import create_supervisor_graph


# =========================================================================
# 1. Schema & Data Contract Tests
# =========================================================================

def test_rag_input_context_schema():
    """Verify RAGInputContext validates required fields and types."""
    payload = RAGInputContext(
        user_query="Need medical help in Cairo",
        retrieved_context="Ambulance service dial 123.",
        candidate_response="Call 123 immediately.",
        domain="emergency"
    )
    assert payload.user_query == "Need medical help in Cairo"
    assert payload.domain == "emergency"
    assert "123" in payload.candidate_response


def test_verification_result_schema():
    """Verify VerificationResult serialization and constraints."""
    result = VerificationResult(
        is_grounded=True,
        has_hallucination=False,
        critical_entities_valid=True,
        confidence_score=0.98,
        hallucination_details=[],
        reasoning="Grounded in official emergency registry.",
        safe_response="Call Tourist Police at 126."
    )
    assert result.is_grounded is True
    assert result.confidence_score == 0.98
    assert result.safe_response == "Call Tourist Police at 126."


# =========================================================================
# 2. Regex & Heuristic Guardrail Unit Tests
# =========================================================================

def test_detect_foreign_emergency_numbers():
    """Verify detection of foreign emergency numbers (911, 999, 112)."""
    assert "911" in detect_foreign_emergency_numbers("In case of crisis, dial 911 for help.")
    assert "999" in detect_foreign_emergency_numbers("Call 999 immediately.")
    assert "112" in detect_foreign_emergency_numbers("Emergency dispatch: 112.")
    assert len(detect_foreign_emergency_numbers("Call 126 for the Tourist Police or 122.")) == 0


def test_extract_phone_numbers():
    """Verify extraction of short codes and phone numbers while ignoring calendar years."""
    text = "In 2026, call 126 for tourist police, 123 for ambulance, and hotline 19654."
    numbers = extract_phone_numbers(text)
    assert "126" in numbers
    assert "123" in numbers
    assert "19654" in numbers
    assert "2026" not in numbers


def test_heuristic_fallback_detects_911():
    """Verify instant rejection and substitution when candidate recommends 911."""
    input_data = RAGInputContext(
        user_query="I was assaulted in Cairo",
        retrieved_context="Tourist Police: 126. Police: 122.",
        candidate_response="I'm so sorry. Please call 911 immediately to reach police dispatch.",
        domain="emergency"
    )
    result = heuristic_verification_fallback(input_data)
    assert result.is_grounded is False
    assert result.has_hallucination is True
    assert result.critical_entities_valid is False
    assert "911" in result.hallucination_details[0]
    assert "126" in result.safe_response


def test_heuristic_fallback_validates_cairo_ambulance():
    """Verify grounded emergency response passes heuristic verification."""
    input_data = RAGInputContext(
        user_query="My friend collapsed, need ambulance in Giza",
        retrieved_context="Ambulance service across Egypt: 123.",
        candidate_response="Please call 123 immediately for emergency medical transport.",
        domain="emergency"
    )
    result = heuristic_verification_fallback(input_data)
    assert result.is_grounded is True
    assert result.has_hallucination is False
    assert result.critical_entities_valid is True
    assert "123" in result.safe_response


def test_heuristic_fallback_flags_unsupported_visa_on_arrival():
    """Verify ungrounded visa-on-arrival promises are rejected with official fallback."""
    input_data = RAGInputContext(
        user_query="Can Nigerian citizens get visa on arrival in Egypt?",
        retrieved_context="Nigerian nationals require a pre-approved consular visa prior to travel.",
        candidate_response="Yes! You can get a visa on arrival at Cairo International Airport for $25.",
        domain="visa"
    )
    result = heuristic_verification_fallback(input_data)
    assert result.is_grounded is False
    assert result.has_hallucination is True
    assert "visa2egypt.gov.eg" in result.safe_response


def test_heuristic_fallback_validates_grounded_advisory():
    """Verify grounded advisory with valid fee and portal passes verification."""
    input_data = RAGInputContext(
        user_query="How much is Egypt e-Visa?",
        retrieved_context="Single-entry e-Visa costs USD 25. Apply at visa2egypt.gov.eg.",
        candidate_response="A single-entry e-Visa costs USD 25. You can apply online.",
        domain="advisories"
    )
    result = heuristic_verification_fallback(input_data)
    assert result.is_grounded is True
    assert result.has_hallucination is False


# =========================================================================
# 3. Verification Guardrail Node Tests
# =========================================================================

def test_help_verification_node_pass_through():
    """Verify LangGraph node preserves grounded response and sets verification flags."""
    state = {
        "user_query": "Need Tourist Police in Luxor",
        "retrieved_context": "Tourist Police nationwide: 126.",
        "candidate_response": "Call the Egyptian Tourist Police directly at 126.",
        "help_domain": "emergency",
        "messages": [("user", "Need Tourist Police in Luxor")]
    }
    updated = help_verification_node(state)
    assert updated["is_verified"] is True
    assert "126" in updated["verified_response"]
    assert updated["response"] == updated["verified_response"]


def test_help_verification_node_replaces_hallucinated_emergency():
    """Verify LangGraph node intercepts hallucinated 911 and replaces it with safe fallback."""
    state = {
        "user_query": "I am being followed in Cairo",
        "retrieved_context": "Tourist Police: 126.",
        "candidate_response": "Run into a shop and call 911 or 999 immediately!",
        "help_domain": "emergency",
        "messages": [("user", "I am being followed in Cairo")]
    }
    updated = help_verification_node(state)
    assert updated["is_verified"] is False
    assert "911" not in updated["verified_response"]
    assert "126" in updated["verified_response"]
    assert len(updated["errors"]) > 0


def test_help_verification_node_replaces_hallucinated_visa():
    """Verify LangGraph node intercepts ungrounded visa on arrival claim."""
    state = {
        "user_query": "Can Indian citizens get visa on arrival?",
        "retrieved_context": "Indian passport holders require consular visas prior to departure.",
        "candidate_response": "No problem! You can easily get a visa on arrival at the airport.",
        "help_domain": "advisories",
        "messages": [("user", "Can Indian citizens get visa on arrival?")]
    }
    updated = help_verification_node(state)
    assert updated["is_verified"] is False
    assert "visa2egypt.gov.eg" in updated["verified_response"]


# =========================================================================
# 4. Coordinate Extraction Tests (Story 2.1)
# =========================================================================

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


# =========================================================================
# 5. Location Resolver Node Tests (Story 2.1)
# =========================================================================

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
    assert result["messages"] == [("user", "What can I do nearby?")]


def test_resolve_location_node_no_coordinates_passthrough():
    state = {
        "messages": [("user", "Hello")],
        "current_location": "Pre-existing Cairo"
    }

    result = resolve_location_node(state)
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


# =========================================================================
# 6. Intent Parsing & Routing Tests (Story 2.1 Item 3)
# =========================================================================

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
    ("طوارئ اريد الشرطة سريعا", "emergency"),
    ("ما هي شروط فيزا مصر للمغاربة؟", "advisories"),
    ("Hello there!", "fallback"),
    ("What is the meaning of life?", "fallback"),
])
def test_parse_tourist_intent_benchmark(query: str, expected_intent: str):
    assert parse_tourist_intent(query) == expected_intent


def test_intent_parser_node():
    state = {"messages": [("user", "Someone stole my wallet, call tourist police!")]}
    updated = intent_parser_node(state)
    assert updated["current_intent"] == "emergency"
    assert updated["help_domain"] == "emergency"


@pytest.mark.parametrize("intent,expected_branch", [
    ("emergency", "emergency"),
    ("emergency_rag", "emergency"),
    ("help_agent", "emergency"),
    ("police", "emergency"),
    ("advisories", "advisories"),
    ("advisories_rag", "advisories"),
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


# =========================================================================
# 7. End-to-End Subgraph Workflows (With Verification Guardrail)
# =========================================================================

def test_tourist_help_subgraph_emergency_flow():
    """Test full execution: emergency query -> RAG candidate -> verification."""
    initial_state = {
        "messages": [("user", "Emergency! I lost my passport and wallet in Cairo, call the police")],
        "user_id": "traveler_101"
    }
    final_state = tourist_help_app.invoke(initial_state)

    assert final_state["help_domain"] == "emergency"
    assert final_state["is_verified"] is True
    assert "126" in final_state["verified_response"]
    assert "122" in final_state["verified_response"]


def test_tourist_help_subgraph_advisories_flow():
    """Test full execution: visa query -> advisory RAG node -> verification."""
    initial_state = {
        "messages": [("user", "What are the Egypt e-Visa rules, costs, and passport requirements?")],
        "user_id": "traveler_202"
    }
    final_state = tourist_help_app.invoke(initial_state)

    assert final_state["help_domain"] == "advisories"
    assert final_state["is_verified"] is True
    assert "e-Visa" in final_state["verified_response"]
    assert "6 months" in final_state["verified_response"]


def test_tourist_help_subgraph_recommendations_flow():
    """Test full execution: recommendations query -> recommendation node -> verification."""
    initial_state = {
        "messages": [("user", "Where can I eat authentic food and visit cool places in Cairo?")],
        "user_id": "traveler_303"
    }
    final_state = tourist_help_app.invoke(initial_state)

    assert final_state["help_domain"] == "recommendations"
    assert final_state["is_verified"] is True
    assert "Koshary" in final_state["verified_response"]


@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_tourist_help_graph_recommendations_with_coordinates(mock_reverse_geocode):
    """Test recommendations with coordinate geocoding + verification."""
    mock_reverse_geocode.return_value = LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Downtown Cairo, Egypt",
        error=None
    )

    initial_state = {
        "messages": [("user", "Can you recommend good places to eat nearby?")],
        "raw_coordinates": {"lat": 30.0444, "lng": 31.2357}
    }
    result = tourist_help_app.invoke(initial_state)

    assert result["current_intent"] == "recommendations"
    assert result["current_location"] == "Downtown Cairo, Egypt"
    assert result["raw_coordinates"] is None
    assert result["is_verified"] is True
    assert "Downtown Cairo" in result["verified_response"]


def test_tourist_help_graph_planner_delegation_branch():
    """Test trip-planning intent delegates to planner."""
    initial_state = {
        "messages": [("user", "Plan a 4-day itinerary for Cairo and Alexandria")]
    }
    result = tourist_help_app.invoke(initial_state)

    assert result["current_intent"] == "plan_trip"
    assert result["next_agent"] == "planner"


def test_tourist_help_graph_fallback_branch():
    """Test ambiguous input routes to fallback node."""
    initial_state = {
        "messages": [("user", "Hello there")]
    }
    result = tourist_help_app.invoke(initial_state)

    assert result["current_intent"] == "fallback"
    assert "help with your trip in Egypt" in result["response"]


# =========================================================================
# 8. Supervisor Graph & API Endpoint Tests
# =========================================================================

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
    assert final_state["current_location"] == "Zamalek, Cairo, Egypt"
    assert final_state["raw_coordinates"] is None
    assert final_state["next_agent"] == "tourist_help"
    assert final_state["is_egypt_scoped"] is True


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


@patch("src.agents.tourist_help.intent_resolver.reverse_geocode")
def test_api_tourist_help_query_endpoint(mock_reverse_geocode):
    from fastapi.testclient import TestClient
    from src.main import app

    mock_reverse_geocode.return_value = LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Tahrir Square, Cairo, Egypt",
        error=None
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/tourist-help/query",
        json={
            "user_id": "test_traveler",
            "message": "Where can I eat authentic food nearby?",
            "location": {"lat": 30.0444, "lng": 31.2357}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["help_domain"] == "recommendations"
    assert data["is_verified"] is True
    assert "Koshary" in data["response"]
    assert data["current_location"] == "Tahrir Square, Cairo, Egypt"
