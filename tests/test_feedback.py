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


# =========================================================================
# Step 2.2: Active Itinerary Modifier Tests (Story 5.2)
# =========================================================================

from src.agents.feedback.itinerary_update import (
    parse_modification_request,
    apply_modification_to_plan,
    validate_plan_through_planner_loops,
    itinerary_modifier_node,
)


def _create_sample_test_plan() -> GeneratedPlan:
    """Helper creating a standard 2-day valid itinerary for tests."""
    day1_activities = [
        {
            "time_slot": "morning",
            "name": "Egyptian Museum in Tahrir",
            "category": "historical",
            "cost_usd": 15.0,
            "duration_hours": 2.5,
            "start_time": "09:30",
            "end_time": "12:00",
            "opening_time": "09:00",
            "closing_time": "17:00",
            "travel_from_prev_mins": 15,
            "accessibility": ["wheelchair access"],
        },
        {
            "time_slot": "afternoon",
            "name": "Khan el-Khalili Bazaar",
            "category": "shopping",
            "cost_usd": 10.0,
            "duration_hours": 2.0,
            "start_time": "13:30",
            "end_time": "15:30",
            "opening_time": "10:00",
            "closing_time": "22:00",
            "travel_from_prev_mins": 20,
            "accessibility": ["wheelchair access", "step-free"],
        },
        {
            "time_slot": "evening",
            "name": "Local Street Food Dinner",
            "category": "dining",
            "cost_usd": 8.0,
            "duration_hours": 1.5,
            "start_time": "19:00",
            "end_time": "20:30",
            "opening_time": "12:00",
            "closing_time": "23:00",
            "travel_from_prev_mins": 15,
            "accessibility": ["wheelchair access"],
        }
    ]

    day2_activities = [
        {
            "time_slot": "morning",
            "name": "Giza Pyramids Complex",
            "category": "sightseeing",
            "cost_usd": 20.0,
            "duration_hours": 3.0,
            "start_time": "09:00",
            "end_time": "12:00",
            "opening_time": "08:00",
            "closing_time": "17:00",
            "travel_from_prev_mins": 30,
            "accessibility": ["wheelchair access"],
        },
        {
            "time_slot": "evening",
            "name": "Nile Corniche Walking Tour",
            "category": "activity",
            "cost_usd": 5.0,
            "duration_hours": 1.5,
            "start_time": "18:00",
            "end_time": "19:30",
            "opening_time": "08:00",
            "closing_time": "22:00",
            "travel_from_prev_mins": 20,
            "accessibility": ["wheelchair access"],
        }
    ]

    return GeneratedPlan(
        plan_id="test_plan_001",
        title="Cairo Highlights",
        days=[
            ItineraryDay(day_number=1, activities=day1_activities, estimated_cost=33.0),
            ItineraryDay(day_number=2, activities=day2_activities, estimated_cost=25.0),
        ],
        status="validated"
    )


def test_parse_modification_request():
    """Tests conversational prompt parsing into PlanModification."""
    plan = _create_sample_test_plan()

    # Italian dinner
    mod1 = parse_modification_request("Change dinner to an Italian restaurant", plan)
    assert mod1.action == "replace"
    assert mod1.time_slot == "evening"
    assert "Italian" in (mod1.replacement_name or "")
    assert mod1.day_number == 1

    # Remove museum on day 2
    mod2 = parse_modification_request("Please remove the museum on day 2", plan)
    assert mod2.action == "remove"
    assert mod2.day_number == 2
    assert mod2.target_place_or_activity == "museum"

    # Add felucca on day 1
    mod3 = parse_modification_request("Add a felucca ride on day 1", plan)
    assert mod3.action == "add"
    assert mod3.day_number == 1
    assert "Felucca" in (mod3.replacement_name or "")


def test_apply_modification_replace_dinner():
    """Tests modifying evening dining to an Italian restaurant."""
    plan = _create_sample_test_plan()
    mod = parse_modification_request("Change dinner to an Italian restaurant", plan)
    updated_plan, summary = apply_modification_to_plan(plan, mod)

    assert "Replaced" in summary
    assert "Italian" in summary
    # Day 1 evening activity should now be Bella Italian
    d1_evening = [a for a in updated_plan.days[0].activities if a.get("time_slot") == "evening"][0]
    assert "Italian" in d1_evening["name"]
    assert d1_evening["cost_usd"] == 18.0
    # Day estimated cost recalculated: 15 + 10 + 18 = 43.0
    assert updated_plan.days[0].estimated_cost == 43.0


def test_apply_modification_remove_activity():
    """Tests removing an activity from a specific day."""
    plan = _create_sample_test_plan()
    mod = parse_modification_request("Please remove Khan el-Khalili on day 1", plan)
    updated_plan, summary = apply_modification_to_plan(plan, mod)

    assert "Removed" in summary
    assert len(updated_plan.days[0].activities) == 2
    # Recalculated cost: 15 + 8 = 23.0
    assert updated_plan.days[0].estimated_cost == 23.0


def test_apply_modification_add_activity():
    """Tests inserting a new activity."""
    plan = _create_sample_test_plan()
    mod = parse_modification_request("Add a felucca ride on day 1", plan)
    updated_plan, summary = apply_modification_to_plan(plan, mod)

    assert "Added" in summary
    assert len(updated_plan.days[0].activities) == 4
    assert any("Felucca" in a["name"] for a in updated_plan.days[0].activities)


def test_validate_plan_through_planner_loops():
    """Ensures modified plan passes Week 3/4 Planner validation loops."""
    plan = _create_sample_test_plan()
    reqs = TripRequirements(destination="Cairo", total_budget=150.0)
    prefs = UserPreferences(
        user_id="test_user",
        name="Test",
        nationality="Egyptian",
        accessibility_needs=["wheelchair access"]
    )

    results = validate_plan_through_planner_loops(plan, reqs, prefs)
    assert results["is_constraint_valid"] is True
    assert results["is_advanced_valid"] is True
    assert results["is_executable"] is True
    assert results["plan_quality"] == "Good"


def test_itinerary_modifier_node_success():
    """Tests full itinerary_modifier_node execution with active plan."""
    plan = _create_sample_test_plan()
    state = {
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo", total_budget=200.0),
        "messages": [("user", "Change dinner to an Italian restaurant")],
        "user_id": "test_user"
    }

    result = itinerary_modifier_node(state)
    assert result["is_executable"] is True
    assert result["next_agent"] == "feedback_completed"
    assert "Itinerary Updated Successfully" in result["response"]
    assert "Italian" in result["response"]
    # Check plan was updated in state
    d1_dinner = [a for a in result["current_plan"].days[0].activities if a.get("time_slot") == "evening"][0]
    assert "Italian" in d1_dinner["name"]


def test_itinerary_modifier_node_no_active_plan():
    """Tests graceful handling when user attempts modifying without an active plan."""
    state = {
        "current_plan": None,
        "messages": [("user", "Change dinner to an Italian restaurant")],
        "user_id": "test_user"
    }

    result = itinerary_modifier_node(state)
    assert "no active itinerary" in result["response"].lower()
    assert result["next_agent"] == "feedback_completed"

