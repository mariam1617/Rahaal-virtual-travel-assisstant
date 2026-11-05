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


