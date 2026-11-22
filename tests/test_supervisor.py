"""
Unit & Benchmark Tests for Supervisor Request Routing (Story 1.2 / Dev 4)
"""

import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

from src.agents.supervisor.router import (
    SupervisorDecision,
    supervisor_router_node,
    supervisor_conditional_edge,
    out_of_scope_responder_node,
    heuristic_fallback_classifier,
    OUT_OF_SCOPE_RESPONSE,
)
from src.agents.supervisor.graph import create_supervisor_graph
from src.agents.feedback.profiling import PROFILING_GREETING_PROMPT
from src.agents.supervisor.context import hydrate_context_node, route_context


# ---------------------------------------------------------
# 1. Schema Validation Tests
# ---------------------------------------------------------
def test_supervisor_decision_schema_valid():
    decision = SupervisorDecision(
        target="planner",
        confidence=0.95,
        is_egypt_scoped=True,
        reasoning="User requested a 3-day Cairo trip."
    )
    assert decision.target == "planner"
    assert decision.confidence == 0.95
    assert decision.is_egypt_scoped is True


def test_supervisor_decision_schema_invalid_confidence():
    with pytest.raises(ValidationError):
        SupervisorDecision(
            target="planner",
            confidence=1.5,  # Exceeds max 1.0
            is_egypt_scoped=True,
            reasoning="Invalid confidence test"
        )


def test_fallback_classifier_does_not_match_substrings():
    assert heuristic_fallback_classifier("Please explain this trip").target == "profiling"
    assert heuristic_fallback_classifier("Tell me about the planet").target == "profiling"
    assert heuristic_fallback_classifier("What is on the balcony?").target == "profiling"


def test_supervisor_router_short_circuits_empty_input(monkeypatch):
    def fail_if_called():
        raise AssertionError("LLM should not be called for empty input")

    monkeypatch.setattr(
        "src.agents.supervisor.router.get_router_llm",
        fail_if_called,
    )
    result = supervisor_router_node({"messages": [("user", "   ")]})

    assert result["next_agent"] == "profiling"
    assert result["confidence"] == 0.0


# ---------------------------------------------------------
# 2. Benchmark Classifier Tests (Across All Intent Categories)
# ---------------------------------------------------------
@pytest.mark.parametrize("query,expected_target,expected_scoped", [
    # Planner
    ("Plan a 3-day itinerary in Cairo on a budget", "planner", True),
    ("I want to visit Luxor and Aswan next month", "planner", True),
    ("Give me a schedule for visiting the Pyramids and Egyptian Museum", "planner", True),

    # Tourist Help
    ("Do US passport holders need a visa on arrival in Egypt?", "tourist_help", True),
    ("What is the emergency number for the Tourist Police in Cairo?", "tourist_help", True),
    ("Where can I eat authentic Koshary near me right now?", "tourist_help", True),
    ("What is the dress code for visiting mosques in Cairo?", "tourist_help", True),

    # Execution
    ("Book a flight ticket to Cairo for next Tuesday", "execution", True),
    ("Reserve my GoBus seats to Hurghada", "execution", True),
    ("Sync my confirmed itinerary to Google Calendar", "execution", True),

    # Feedback
    ("Please change my plan and remove the museum on day 2", "feedback", True),
    ("I want to modify the hotel in Alexandria", "feedback", True),

    # Out of Scope (Geographic Exclusion)
    ("Plan a 4-day romantic vacation to Paris", "out_of_scope", False),
    ("Can you find me top hotels in Dubai?", "out_of_scope", False),
    ("I want to visit Tokyo and Kyoto for cherry blossom season", "out_of_scope", False),
    ("What is the weather in London this weekend?", "out_of_scope", False),

    # Ambiguous / Low Confidence (Fallback to Profiling)
    ("Hello", "profiling", True),
    ("Hi there", "profiling", True),
    ("Help me", "profiling", True),
])
def test_supervisor_classification_benchmark(query: str, expected_target: str, expected_scoped: bool):
    decision = heuristic_fallback_classifier(query)
    assert decision.target == expected_target, f"Failed for query: '{query}' -> got {decision.target}, expected {expected_target}"
    assert decision.is_egypt_scoped == expected_scoped, f"Scope mismatch for query: '{query}'"


# ---------------------------------------------------------
# 3. Conditional Edge Logic Tests
# ---------------------------------------------------------
def test_conditional_edge_out_of_scope():
    state = {
        "messages": [("user", "Plan a trip to Paris")],
        "next_agent": "out_of_scope",
        "is_egypt_scoped": False,
        "confidence": 1.0
    }
    assert supervisor_conditional_edge(state) == "out_of_scope"


def test_conditional_edge_low_confidence_fallback():
    # The router node owns low-confidence fallback and stores profiling as next_agent.
    state = {
        "messages": [("user", "Somewhere nice")],
        "next_agent": "profiling",
        # "next_agent": "planner",
        "is_egypt_scoped": True,
        "confidence": 0.50
    }
    assert supervisor_conditional_edge(state) == "profiling"


def test_conditional_edge_valid_subgraph():
    state = {
        "messages": [("user", "3 days in Cairo")],
        "next_agent": "planner",
        "is_egypt_scoped": True,
        "confidence": 0.95
    }
    assert supervisor_conditional_edge(state) == "planner"


def test_supervisor_router_records_low_confidence_adjustment(monkeypatch):
    monkeypatch.setenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setattr(
        "src.agents.supervisor.router.get_router_llm",
        lambda: None,
    )

    result = supervisor_router_node({
        "messages": [("user", "Hello")],
    })

    assert result["next_agent"] == "profiling"
    assert "Low confidence" in result["routing_reason"]


def test_supervisor_hard_handoff_cap_returns_finish():
    result = supervisor_router_node({
        "messages": [("user", "Plan a trip")],
        "completed_steps": ["planner"],
        "handoff_count": 4,
        "max_handoff_count": 4,
    })

    assert result["next_agent"] == "FINISH"


def test_supervisor_does_not_repeat_completed_subgraph(monkeypatch):
    monkeypatch.setattr(
        "src.agents.supervisor.router.get_router_llm",
        lambda: None,
    )
    result = supervisor_router_node({
        "messages": [("user", "Plan a 3-day Cairo itinerary")],
        "completed_steps": ["planner"],
        "handoff_count": 1,
        "max_handoff_count": 4,
    })

    assert result["next_agent"] == "FINISH"


def test_context_hydration_known_user():
    """Verify known user preferences are loaded before routing."""
    result = hydrate_context_node({"user_id": "user_101", "current_location": "Cairo"})

    assert result["context_loaded"] is True
    assert result["user_preferences"].user_id == "user_101"
    assert result["current_location"] == "Cairo"
    assert route_context(result) == "supervisor"


def test_context_hydration_unknown_user_routes_to_profiling():
    """Verify unknown users do not silently receive an empty profile."""
    result = hydrate_context_node({"user_id": "unknown_user"})

    assert result["context_loaded"] is False
    assert result["needs_profiling"] is True
    assert "No profile found" in result["context_warning"]
    assert route_context(result) == "profiling"


def test_supervisor_graph_unknown_user_enters_profiling():
    """Verify unknown-user context gating occurs before Supervisor classification."""
    graph = create_supervisor_graph()
    result = graph.invoke({
        "user_id": "unknown_user",
        "messages": [("user", "Plan a 3-day itinerary for Cairo")],
    })

    assert result["needs_profiling"] is True
    assert result["next_agent"] == "profiling_completed"


# ---------------------------------------------------------
# 4. End-to-End LangGraph Routing Tests
# ---------------------------------------------------------
def test_graph_routing_out_of_scope_flow():
    graph = create_supervisor_graph()
    initial_state = {
        "messages": [("user", "I want to visit the Eiffel Tower in Paris")]
    }
    final_state = graph.invoke(initial_state)

    assert final_state["is_egypt_scoped"] is False
    assert final_state["next_agent"] == "END"
    assert OUT_OF_SCOPE_RESPONSE in final_state["response"]


def test_graph_routing_ambiguous_fallback_flow():
    graph = create_supervisor_graph()
    initial_state = {
        "messages": [("user", "Hello!")]
    }
    final_state = graph.invoke(initial_state)

    assert final_state["next_agent"] == "profiling_completed"
    assert PROFILING_GREETING_PROMPT in final_state["response"]


def test_graph_routing_valid_planner_flow():
    graph = create_supervisor_graph()
    initial_state = {
        "messages": [("user", "Plan a 3-day itinerary for Cairo")]
    }
    final_state = graph.invoke(initial_state)

    assert final_state["is_egypt_scoped"] is True
    assert final_state["next_agent"] == "planner"
    assert final_state["confidence"] >= 0.65
