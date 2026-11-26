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


# =========================================================================
# Step 2.3: Insight Extractor Pipeline Tests (Story 5.3)
# =========================================================================

from src.agents.feedback.preference_learning import (
    compute_insight_embedding,
    extract_preference_insights,
    insight_extractor_node,
    EMBEDDING_DIM,
)
import math


def test_compute_insight_embedding():
    """Validates embedding dimensionality, normalization, and determinism."""
    vec1 = compute_insight_embedding("I prefer morning flights")
    vec2 = compute_insight_embedding("I prefer morning flights")
    vec3 = compute_insight_embedding("I am vegan and eat halal")

    assert len(vec1) == EMBEDDING_DIM
    # Determinism
    assert vec1 == vec2
    # Semantics difference
    assert vec1 != vec3
    # L2 Unit Normalization
    norm = math.sqrt(sum(x * x for x in vec1))
    assert abs(norm - 1.0) < 1e-3


def test_extract_preference_insights_flight_habit():
    """Tests extracting flight timing preference."""
    pref = extract_preference_insights("I prefer morning flights and direct flights")
    assert "morning flights" in pref.travel_habits
    assert "direct flights" in pref.travel_habits
    assert any(i.key == "morning_flights" for i in pref.insights)
    # Check embedding is present
    flight_insight = [i for i in pref.insights if i.key == "morning_flights"][0]
    assert len(flight_insight.embedding) == EMBEDDING_DIM


def test_extract_preference_insights_dietary_and_accessibility():
    """Tests extracting dietary rules and accessibility needs."""
    pref = extract_preference_insights("I am vegan, eat halal, and need wheelchair access")
    assert DietaryPreference.VEGAN in pref.dietary
    assert DietaryPreference.HALAL in pref.dietary
    assert "wheelchair access" in pref.accessibility_needs
    assert len(pref.insights) >= 3


def test_extract_preference_insights_luxury_and_interests():
    """Tests extracting luxury tier and passions."""
    pref = extract_preference_insights("I always prefer luxury 5-star hotels and love historical sites")
    assert pref.budget_level == "luxury"
    assert "historical sites" in pref.interests
    assert any("luxury" in tag for i in pref.insights for tag in i.tags)


def test_insight_extractor_node():
    """Tests the insight_extractor_node updates state and routes to ltm_updater."""
    state = {
        "messages": [("user", "I prefer morning flights and boutique hotels")],
        "user_id": "test_user"
    }
    result = insight_extractor_node(state)

    assert result["next_agent"] == "ltm_updater"
    assert "extracted_insights" in result
    assert len(result["extracted_insights"]) >= 1
    assert "morning_flights" in [i["key"] for i in result["extracted_insights"]]
    assert result["pending_preferences"]["travel_habits"] == ["morning flights"]


# =========================================================================
# Step 2.4: Long-Term Preference Update & Profile Store Tests (Story 5.3)
# =========================================================================

from src.db.profile_store import ProfileStore, cosine_similarity
from src.db.repositories.user_repo import SQLiteUserRepository, UserRecord
from src.agents.feedback.preference_learning import ltm_preference_update_node
from src.agents.feedback.graph import feedback_app, create_feedback_graph


def _ensure_test_user(user_id: str, name: str = "Test User"):
    repo = SQLiteUserRepository()
    if not repo.get_by_user_id(user_id):
        repo.create(UserRecord(
            user_id=user_id,
            username=user_id,
            name=name,
            hashed_password="hashed_test_password"
        ))


def test_profile_store_get_and_update():
    """Validates persistent profile reading and updating in LTM store."""
    user_id = "test_user_ltm"
    _ensure_test_user(user_id, name="LTM Tester")

    profile = ProfileStore.ensure_user_profile(user_id, name="LTM Tester")
    assert profile.user_id == user_id

    # Update with extracted preferences
    extracted = extract_preference_insights("I am vegan, prefer morning flights, and love museums")
    updated = ProfileStore.update_user_preferences(user_id, extracted)

    assert DietaryPreference.VEGAN in updated.dietary
    assert "museums" in updated.interests
    assert "morning flights" in updated.travel_habits


def test_profile_store_vector_insight_search():
    """Validates semantic search over historical preference insights."""
    user_id = "search_user_test"
    _ensure_test_user(user_id)
    extracted = extract_preference_insights("I prefer morning flights and luxury 5-star hotels")
    ProfileStore.update_user_preferences(user_id, extracted)

    # Query with semantic variation
    results = ProfileStore.search_user_insights(user_id, query="early flight schedule departure", top_k=2)
    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["key"] == "morning_flights"
    assert top_hit["similarity_score"] > 0.0


def test_ltm_preference_update_node_standalone():
    """Tests the ltm_preference_update_node execution for a pure preference feedback."""
    user_id = "node_user_test"
    _ensure_test_user(user_id)
    state = {
        "user_id": user_id,
        "messages": [("user", "I prefer morning flights and eat halal")],
        "feedback_type": "long_term_preference"
    }

    result = ltm_preference_update_node(state)
    assert result["next_agent"] == "feedback_completed"
    assert "Profile Updated" in result["response"]
    assert DietaryPreference.HALAL in result["user_preferences"].dietary


def test_ltm_preference_update_node_compound_chains_to_modifier():
    """Tests compound intent routes to itinerary_modifier after saving to LTM."""
    user_id = "compound_user_test"
    _ensure_test_user(user_id)
    state = {
        "user_id": user_id,
        "messages": [("user", "I am vegetarian, please change dinner to a vegetarian restaurant")],
        "feedback_type": "both"
    }

    result = ltm_preference_update_node(state)
    assert result["next_agent"] == "itinerary_modifier"
    assert DietaryPreference.VEGETARIAN in result["user_preferences"].dietary


# =========================================================================
# End-to-End Feedback Subgraph Workflow Tests
# =========================================================================

def test_feedback_subgraph_e2e_trip_adjustment():
    """End-to-end execution of trip modification request through feedback_app."""
    plan = _create_sample_test_plan()
    input_state = {
        "user_id": "e2e_user_trip",
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo", total_budget=300.0),
        "messages": [("user", "Change dinner to an Italian restaurant")]
    }

    result = feedback_app.invoke(
        input_state,
        config={"configurable": {"thread_id": "test_e2e_trip_adj"}}
    )

    assert result["feedback_type"] == "trip_adjustment"
    assert "Italian" in result["response"]
    d1_dinner = [a for a in result["current_plan"].days[0].activities if a.get("time_slot") == "evening"][0]
    assert "Italian" in d1_dinner["name"]


def test_feedback_subgraph_e2e_preference_learning():
    """End-to-end execution of preference learning request through feedback_app."""
    _ensure_test_user("e2e_user_pref")
    input_state = {
        "user_id": "e2e_user_pref",
        "messages": [("user", "I prefer morning flights and boutique hotels")]
    }

    result = feedback_app.invoke(
        input_state,
        config={"configurable": {"thread_id": "test_e2e_pref_learn"}}
    )

    assert result["feedback_type"] == "long_term_preference"
    assert "morning flights" in result["response"].lower()
    assert "morning flights" in result["user_preferences"].travel_habits


def test_feedback_subgraph_e2e_compound():
    """End-to-end execution of compound request (both preference + trip adjustment)."""
    _ensure_test_user("e2e_user_compound")
    plan = _create_sample_test_plan()
    input_state = {
        "user_id": "e2e_user_compound",
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo", total_budget=300.0),
        "messages": [("user", "I am vegetarian, so please change tonight's dinner to a vegetarian restaurant")]
    }

    result = feedback_app.invoke(
        input_state,
        config={"configurable": {"thread_id": "test_e2e_compound"}}
    )

    assert result["feedback_type"] == "both"
    # Preference was saved to LTM
    assert DietaryPreference.VEGETARIAN in result["user_preferences"].dietary
    # Itinerary dinner was modified
    d1_dinner = [a for a in result["current_plan"].days[0].activities if a.get("time_slot") == "evening"][0]
    assert "Vegetarian" in d1_dinner["name"] or "Abou Tarek" in d1_dinner["name"]


def test_feedback_subgraph_e2e_fallback():
    """End-to-end execution of unrecognized feedback query."""
    input_state = {
        "user_id": "e2e_user_fallback",
        "messages": [("user", "Hello there")]
    }

    result = feedback_app.invoke(
        input_state,
        config={"configurable": {"thread_id": "test_e2e_fallback"}}
    )

    assert result["feedback_type"] == "unknown"
    assert "couldn't quite understand" in result["response"].lower()


# =========================================================================
# FastAPI Feedback Endpoint Tests
# =========================================================================

from fastapi.testclient import TestClient
from src.main import app


def test_api_feedback_trip_adjustment():
    """Tests POST /api/v1/feedback/process for trip adjustment."""
    client = TestClient(app)
    plan = _create_sample_test_plan()

    payload = {
        "user_id": "api_feedback_user",
        "message": "Change dinner to an Italian restaurant",
        "current_plan": plan.model_dump(),
        "trip_requirements": {"destination": "Cairo", "total_budget": 500.0}
    }

    response = client.post("/api/v1/feedback/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "trip_adjustment"
    assert "Italian" in data["response"]
    assert data["current_plan"] is not None


def test_api_feedback_preference_learning():
    """Tests POST /api/v1/feedback/process for preference learning."""
    client = TestClient(app)

    payload = {
        "user_id": "api_pref_user",
        "message": "I prefer morning flights and luxury hotels",
    }

    response = client.post("/api/v1/feedback/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feedback_type"] == "long_term_preference"
    assert "Profile Updated" in data["response"]
    assert data["user_preferences"] is not None




