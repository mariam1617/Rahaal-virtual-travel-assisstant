"""
Unit and Integration Tests for Feedback & Memory Subgraph (Epic 5)
Story 5.2: Feedback Router & Active Itinerary Modifier
Story 5.3: Insight Extractor & Long-Term Preference Learning
"""

import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

from src.state.schema import (
    FeedbackClassification,
    FeedbackType,
    GeneratedPlan,
    ItineraryDay,
    TripRequirements,
    UserPreferences,
    DietaryPreference,
)
from src.agents.feedback.router import (
    heuristic_feedback_classifier,
    feedback_router_node,
    route_feedback,
)


# =========================================================================
# Step 2.1: Feedback Router Tests (Story 5.2, Story 5.3)
# =========================================================================

def test_feedback_classification_schema_valid():
    """Validates FeedbackClassification schema fields."""
    decision = FeedbackClassification(
        feedback_type="trip_adjustment",
        confidence=0.92,
        reasoning="User wants Italian dinner instead of current plan.",
        modification_intent="change dinner to an Italian restaurant"
    )
    assert decision.feedback_type == "trip_adjustment"
    assert decision.confidence == 0.92
    assert "Italian" in decision.modification_intent


def test_feedback_classification_schema_invalid_confidence():
    """Ensures confidence bounds [0.0, 1.0] are strictly enforced."""
    with pytest.raises(ValidationError):
        FeedbackClassification(
            feedback_type="trip_adjustment",
            confidence=1.8,
            reasoning="Invalid confidence test"
        )


@pytest.mark.parametrize("query,expected_type", [
    # 1. Trip Adjustments (Story 5.2)
    ("Change dinner to an Italian restaurant", "trip_adjustment"),
    ("Please remove the museum on day 2", "trip_adjustment"),
    ("Can we swap the afternoon activity for a felucca ride on day 1?", "trip_adjustment"),
    ("Make the morning tour start earlier", "trip_adjustment"),
    ("Replace the hotel in Alexandria with another one", "trip_adjustment"),
    ("Cancel tomorrow evening dinner and give us free time", "trip_adjustment"),

    # 2. Long-Term Preferences (Story 5.3)
    ("I prefer morning flights", "long_term_preference"),
    ("I am vegan", "long_term_preference"),
    ("I'm vegetarian and eat halal only", "long_term_preference"),
    ("I always prefer luxury 5-star hotels", "long_term_preference"),
    ("I love historical sites and museums", "long_term_preference"),
    ("Keep in mind that I require wheelchair accessibility for future trips", "long_term_preference"),

    # 3. Compound / Both (Story 5.2 + 5.3)
    ("I am vegetarian, so please change tonight's dinner to a vegetarian restaurant", "both"),
    ("I prefer morning flights, can we switch tomorrow's flight to 8 AM?", "both"),

    # 4. Ambiguous / Unknown Fallback
    ("hello there", "unknown"),
    ("", "unknown"),
])
def test_heuristic_feedback_classifier_benchmark(query: str, expected_type: str):
    """Benchmarks the deterministic classifier across all categories."""
    decision = heuristic_feedback_classifier(query)
    assert decision.feedback_type == expected_type, (
        f"Query: '{query}' -> Expected {expected_type}, got {decision.feedback_type} "
        f"(reasoning: {decision.reasoning})"
    )
    if expected_type != "unknown":
        assert decision.confidence >= 0.80


def test_feedback_router_node_trip_adjustment():
    """Verifies node execution for a trip adjustment."""
    state = {
        "messages": [("user", "Change dinner to an Italian restaurant")],
        "user_id": "test_user"
    }
    result = feedback_router_node(state)
    assert result["feedback_type"] == "trip_adjustment"
    assert result["next_agent"] == "itinerary_modifier"
    assert result["feedback_confidence"] >= 0.85


def test_feedback_router_node_long_term_preference():
    """Verifies node execution for a long-term preference."""
    state = {
        "messages": [HumanMessage(content="I prefer morning flights")],
        "user_id": "test_user"
    }
    result = feedback_router_node(state)
    assert result["feedback_type"] == "long_term_preference"
    assert result["next_agent"] == "insight_extractor"
    assert result["feedback_confidence"] >= 0.85


def test_feedback_router_node_compound():
    """Verifies node execution for a compound intent."""
    state = {
        "messages": [("user", "I'm vegetarian, please change dinner to a vegetarian restaurant")],
        "user_id": "test_user"
    }
    result = feedback_router_node(state)
    assert result["feedback_type"] == "both"
    assert result["next_agent"] == "insight_extractor"


def test_route_feedback_conditional_edge():
    """Tests the LangGraph conditional edge routing function."""
    assert route_feedback({"feedback_type": "trip_adjustment"}) == "itinerary_modifier"
    assert route_feedback({"feedback_type": "long_term_preference"}) == "insight_extractor"
    assert route_feedback({"feedback_type": "both"}) == "insight_extractor"
    assert route_feedback({"feedback_type": "unknown"}) == "feedback_fallback"
