"""
Unit & Guardrail Tests for Tourist Help Verification Node  
Tests hallucination detection, emergency number validation, visa grounding, and LangGraph state updates.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.tourist_help.schemas import (
    RAGInputContext,
    VerificationResult
)
from src.agents.tourist_help.prompts import (
    OFFICIAL_EGYPT_EMERGENCY_NUMBERS,
    DEFAULT_EMERGENCY_FALLBACK,
    DEFAULT_ADVISORY_FALLBACK,
    DEFAULT_GENERAL_FALLBACK,
    HELP_VERIFICATION_SYSTEM_PROMPT
)
from src.agents.tourist_help.verification import (
    detect_foreign_emergency_numbers,
    extract_phone_numbers,
    heuristic_verification_fallback,
    verify_rag_response,
    help_verification_node
)


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
        reasoning="All numbers verified against context.",
        safe_response="Call 123 for ambulance."
    )
    dumped = result.model_dump()
    assert dumped["is_grounded"] is True
    assert dumped["has_hallucination"] is False
    assert dumped["confidence_score"] == 0.98


# =========================================================================
# 2. Fast Deterministic Regex & Foreign Number Guard Tests
# =========================================================================

@pytest.mark.parametrize("foreign_number", ["911", "999", "112"])
def test_detect_foreign_emergency_numbers(foreign_number):
    """Ensure non-Egyptian emergency numbers (911, 999, 112) are detected."""
    sample_text = f"Stay calm and call {foreign_number} immediately for police dispatch."
    detected = detect_foreign_emergency_numbers(sample_text)
    assert foreign_number in detected


def test_no_false_positive_on_egyptian_numbers():
    """Ensure valid Egyptian emergency numbers are NOT flagged as foreign."""
    egypt_text = "Call 122 for police, 123 for ambulance, 126 for tourist police, and 180 for fire."
    detected = detect_foreign_emergency_numbers(egypt_text)
    assert len(detected) == 0


def test_extract_phone_numbers():
    """Verify Egyptian short codes and numbers are properly extracted."""
    text = "Call Tourist Police at 126 or general line 01221110000 or hotline 19654."
    numbers = extract_phone_numbers(text)
    assert "126" in numbers
    assert "01221110000" in numbers
    assert "19654" in numbers


# =========================================================================
# 3. Emergency Assistance Guardrail Tests (Story 2.2)
# =========================================================================

def test_emergency_verification_passes_grounded_response():
    """Grounded emergency advice with correct numbers must pass verification."""
    payload = RAGInputContext(
        user_query="My friend was robbed near the Pyramids. Who do I call?",
        retrieved_context="Tourist Police: 126 for tourist safety, harassment, and theft. General Police: 122.",
        candidate_response="Please contact the Tourist Police at 126 immediately, or reach General Police at 122.",
        domain="emergency"
    )
    result = verify_rag_response(payload)
    assert result.is_grounded is True
    assert result.has_hallucination is False
    assert result.critical_entities_valid is True
    assert "126" in result.safe_response


def test_emergency_verification_intercepts_hallucinated_911():
    """Any candidate recommending '911' in Egypt must be immediately intercepted and sanitized."""
    payload = RAGInputContext(
        user_query="There is an emergency in downtown Cairo!",
        retrieved_context="Ambulance: 123, Police: 122",
        candidate_response="Call 911 right now to report the incident to emergency services!",
        domain="emergency"
    )
    result = verify_rag_response(payload)
    assert result.is_grounded is False
    assert result.has_hallucination is True
    assert result.critical_entities_valid is False
    # Ensure safe fallback provides verified Egyptian numbers
    assert "126" in result.safe_response or "122" in result.safe_response or "123" in result.safe_response
    assert "911" not in result.safe_response


def test_emergency_verification_intercepts_fabricated_local_number():
    """An invented non-directory phone number for police must be intercepted."""
    payload = RAGInputContext(
        user_query="Need police assistance",
        retrieved_context="General Police: 122",
        candidate_response="You can call the local precinct hotline at 88887766.",
        domain="emergency"
    )
    result = verify_rag_response(payload)
    assert result.has_hallucination is True
    assert result.critical_entities_valid is False
    assert result.safe_response == DEFAULT_EMERGENCY_FALLBACK


# =========================================================================
# 4. Visa & Tourist Advisory Guardrail Tests (Story 2.3)
# =========================================================================

def test_visa_verification_passes_grounded_response():
    """Accurate visa advice matching retrieved rules must pass."""
    context = (
        "Eligible countries for Egypt e-Visa include Germany, France, United States, United Kingdom. "
        "Single-entry e-Visa costs USD 25. Passports must have at least 6 months validity."
    )
    payload = RAGInputContext(
        user_query="Can a German citizen get an e-Visa and what does it cost?",
        retrieved_context=context,
        candidate_response="Yes, German citizens are eligible for an Egypt e-Visa. A single-entry visa costs USD 25, and your passport must be valid for 6 months.",
        domain="visa"
    )
    result = verify_rag_response(payload)
    assert result.is_grounded is True
    assert result.has_hallucination is False
    assert result.safe_response == payload.candidate_response


def test_visa_verification_intercepts_hallucinated_visa_on_arrival():
    """Promising ungrounded 'visa on arrival' when not supported by context must be intercepted."""
    context = (
        "Citizens of countries not on the e-Visa list must obtain a consular visa prior to departure from an Egyptian Embassy."
    )
    payload = RAGInputContext(
        user_query="Can I get a visa on arrival with my passport?",
        retrieved_context=context,
        candidate_response="Don't worry, you can easily obtain a visa on arrival at Cairo International Airport upon landing!",
        domain="visa"
    )
    result = verify_rag_response(payload)
    assert result.has_hallucination is True
    assert result.is_grounded is False
    assert result.safe_response == DEFAULT_ADVISORY_FALLBACK


# =========================================================================
# 5. Empty Context & Edge Case Tests
# =========================================================================

def test_empty_context_with_unverified_claims():
    """If retrieved context is empty and candidate asserts random phone numbers, fallback is triggered."""
    payload = RAGInputContext(
        user_query="What is the direct number for Sharm El Sheikh marine rescue?",
        retrieved_context="",
        candidate_response="Call marine rescue at 77712345.",
        domain="emergency"
    )
    result = verify_rag_response(payload)
    assert result.has_hallucination is True
    assert result.is_grounded is False
    assert result.safe_response == DEFAULT_EMERGENCY_FALLBACK


# =========================================================================
# 6. LangGraph Node Integration Tests
# =========================================================================

def test_help_verification_node_grounded_state():
    """Verify help_verification_node updates state correctly for a grounded answer."""
    initial_state = {
        "user_query": "I feel sick in Luxor",
        "retrieved_context": "Ambulance: 123",
        "candidate_response": "Please dial 123 for ambulance services in Egypt.",
        "help_domain": "emergency",
        "messages": [HumanMessage(content="I feel sick in Luxor")],
        "errors": []
    }
    updated_state = help_verification_node(initial_state)

    assert updated_state["is_verified"] is True
    assert updated_state["verified_response"] == "Please dial 123 for ambulance services in Egypt."
    assert len(updated_state["messages"]) == 2
    assert isinstance(updated_state["messages"][-1], AIMessage)
    assert updated_state["next_agent"] == "completed"


def test_help_verification_node_intercepts_hallucination():
    """Verify help_verification_node replaces hallucinated content with safe fallback in state."""
    initial_state = {
        "user_query": "I need help!",
        "retrieved_context": "Ambulance: 123",
        "candidate_response": "Call 911 immediately!",
        "help_domain": "emergency",
        "messages": [HumanMessage(content="I need help!")],
        "errors": []
    }
    updated_state = help_verification_node(initial_state)

    assert updated_state["is_verified"] is False
    assert "911" not in updated_state["verified_response"]
    assert "126" in updated_state["verified_response"] or "122" in updated_state["verified_response"]
    assert len(updated_state["errors"]) > 0
    assert "Guardrail intercepted response" in updated_state["errors"][0]


# =========================================================================
# 7. Mock LLM Structured Output Test
# =========================================================================

def test_verify_rag_response_with_mock_llm():
    """Verify verify_rag_response handles LLM structured output invocation."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    mock_result = VerificationResult(
        is_grounded=True,
        has_hallucination=False,
        critical_entities_valid=True,
        confidence_score=0.99,
        hallucination_details=[],
        reasoning="Mock LLM verified all facts against context.",
        safe_response="Call Tourist Police at 126."
    )
    mock_structured_llm.invoke.return_value = mock_result

    payload = RAGInputContext(
        user_query="Lost passport in Cairo",
        retrieved_context="Tourist Police: 126",
        candidate_response="Call Tourist Police at 126.",
        domain="emergency"
    )

    result = verify_rag_response(payload, llm=mock_llm)
    assert result.is_grounded is True
    assert result.confidence_score == 0.99
    assert result.safe_response == "Call Tourist Police at 126."
    mock_structured_llm.invoke.assert_called_once()


# =========================================================================
# 8. Tourist Help Subgraph End-to-End Tests (Story 2.1 - 2.4 Integration)
# =========================================================================

from src.agents.tourist_help.graph import tourist_help_app, create_tourist_help_graph


def test_tourist_help_subgraph_emergency_flow():
    """Test full execution: emergency query -> emergency RAG -> verification -> verified output."""
    initial_state = {
        "messages": [("user", "Emergency! I need an ambulance and police in Cairo right now.")],
        "user_id": "traveler_101"
    }
    final_state = tourist_help_app.invoke(initial_state)

    assert final_state["help_domain"] == "emergency"
    assert final_state["is_verified"] is True
    assert "126" in final_state["verified_response"]
    assert "123" in final_state["verified_response"]
    assert final_state["next_agent"] == "completed"


def test_tourist_help_subgraph_visa_flow():
    """Test full execution: visa inquiry -> advisory RAG -> verification -> verified output."""
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


