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


# =========================================================================
# 2. Milestone 2: Active Profiling Tests (Story 5.1)
# =========================================================================
from src.agents.feedback.profiling import (
    extract_constraints_from_text,
    detect_missing_constraints,
    ask_user_questions,
    update_session_preferences,
    profiling_check_node,
)
from src.state.schema import TripRequirements


def test_extract_constraints_from_text():
    """Verify extraction of destination, duration, and budget from user prompt."""
    text = "I want to plan a 3-day trip to Luxor with a budget of $450"
    extracted = extract_constraints_from_text(text)
    assert extracted["destination"] == "Luxor"
    assert extracted["duration_days"] == 3
    assert extracted["total_budget"] == 450.0


def test_detect_missing_constraints_all_missing():
    """Verify detection when no requirements are provided."""
    missing = detect_missing_constraints(None)
    assert "destination" in missing
    assert "duration" in missing
    assert "budget" in missing


def test_detect_missing_constraints_partial():
    """Verify detection when only destination is set."""
    req = TripRequirements(destination="Cairo")
    missing = detect_missing_constraints(req)
    assert "destination" not in missing
    assert "duration" in missing
    assert "budget" in missing


def test_ask_user_questions():
    """Verify dynamic question generation for missing parameters."""
    question = ask_user_questions(["destination", "budget"])
    assert "city or region" in question
    assert "budget" in question


def test_update_session_preferences():
    """Verify updating session requirements."""
    state = {}
    updated = update_session_preferences(state, {"destination": "Aswan", "total_budget": 600.0, "duration_days": 4})
    req = updated["trip_requirements"]
    assert req.destination == "Aswan"
    assert req.total_budget == 600.0
    assert getattr(req, "duration_days", None) == 4


def test_profiling_check_node_incomplete():
    """Verify profiling node triggers clarifying questions when query is vague."""
    state = {"messages": [("user", "I want to travel")]}
    res = profiling_check_node(state)
    assert res["needs_profiling"] is True
    assert res["next_agent"] == "profiling"
    assert len(res["missing_constraints"]) > 0


def test_profiling_check_node_complete():
    """Verify profiling node advances to draft_generator when all constraints are present."""
    state = {"messages": [("user", "Plan 4 days in Cairo with budget $500")]}
    res = profiling_check_node(state)
    assert res["needs_profiling"] is False
    assert res["next_agent"] == "draft_generator"


# =========================================================================
# 3. Milestone 3: Draft Generator Tests (Story 3.1)
# =========================================================================
from src.agents.planner.draft_generator import (
    build_draft_itinerary,
    draft_generator_node,
)
from src.state.schema import GeneratedPlan


def test_build_draft_itinerary_structure():
    """Verify draft itinerary builds valid days and time slots."""
    req = TripRequirements(destination="Cairo", duration_days=3, total_budget=400.0)
    plan = build_draft_itinerary(req)

    assert isinstance(plan, GeneratedPlan)
    assert plan.status == "draft"
    assert len(plan.days) == 3
    assert "Cairo" in plan.title

    for day in plan.days:
        assert len(day.activities) >= 2
        assert day.estimated_cost > 0
        slots = [a["time_slot"] for a in day.activities]
        assert "morning" in slots


def test_draft_generator_node():
    """Verify draft generator node updates state with current_plan and routes to validator."""
    state = {
        "trip_requirements": TripRequirements(destination="Luxor", duration_days=2, total_budget=300.0),
        "messages": [("user", "Plan 2 days in Luxor")]
    }
    result = draft_generator_node(state)

    assert "current_plan" in result
    plan = result["current_plan"]
    assert len(plan.days) == 2
    assert result["next_agent"] == "constraint_validator"
    assert "Luxor" in result["response"]


# =========================================================================
# 4. Milestone 4: Constraint & Budget Validator Tests (Story 3.2)
# =========================================================================
from src.agents.planner.constraint_validator import (
    validate_budget,
    validate_logistics,
    constraint_validator_node,
)
from src.state.schema import ItineraryDay


def test_validate_budget_within_budget():
    """Verify budget validation passes when total cost is within budget."""
    days = [
        ItineraryDay(day_number=1, activities=[], estimated_cost=80.0),
        ItineraryDay(day_number=2, activities=[], estimated_cost=90.0),
    ]
    plan = GeneratedPlan(plan_id="p1", title="Trip", days=days)
    is_valid, err, total_cost, overage = validate_budget(plan, max_budget=200.0)
    assert is_valid is True
    assert err is None
    assert total_cost == 170.0
    assert overage == 0.0


def test_validate_budget_exceeds():
    """Verify budget validation fails and calculates overage when cost exceeds budget."""
    days = [
        ItineraryDay(day_number=1, activities=[], estimated_cost=150.0),
        ItineraryDay(day_number=2, activities=[], estimated_cost=150.0),
    ]
    plan = GeneratedPlan(plan_id="p2", title="Trip", days=days)
    is_valid, err, total_cost, overage = validate_budget(plan, max_budget=200.0)
    assert is_valid is False
    assert "exceeds user budget" in err
    assert total_cost == 300.0
    assert overage == 100.0


def test_validate_logistics_normal():
    """Verify logistical transit validation passes for reasonable durations."""
    activities = [
        {"name": "Museum", "travel_from_prev_mins": 30},
        {"name": "Pyramids", "travel_from_prev_mins": 45},
    ]
    plan = GeneratedPlan(plan_id="p3", days=[ItineraryDay(day_number=1, activities=activities)])
    is_valid, errors = validate_logistics(plan)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_logistics_exceeds():
    """Verify logistical transit validation catches excessive travel time."""
    activities = [
        {"name": "Alexandria", "travel_from_prev_mins": 180},
        {"name": "Cairo", "travel_from_prev_mins": 180},
    ]
    plan = GeneratedPlan(plan_id="p4", days=[ItineraryDay(day_number=1, activities=activities)])
    is_valid, errors = validate_logistics(plan, max_transit_hours=3.5)
    assert is_valid is False
    assert len(errors) == 1
    assert "Day 1 transit time" in errors[0]


def test_constraint_validator_node_valid():
    """Verify constraint validator node approves valid plan."""
    req = TripRequirements(destination="Cairo", total_budget=500.0, duration_days=2)
    plan = build_draft_itinerary(req)
    state = {"current_plan": plan, "trip_requirements": req, "repair_attempts": 0}

    result = constraint_validator_node(state)
    assert result["is_constraint_valid"] is True
    assert result["next_agent"] == "plan_validated"
    assert result["current_plan"].status == "validated"


def test_constraint_validator_node_invalid():
    """Verify constraint validator node detects overage and routes to repair."""
    req = TripRequirements(destination="Cairo", total_budget=20.0, duration_days=3)  # Insufficient budget
    plan = build_draft_itinerary(req)
    state = {"current_plan": plan, "trip_requirements": req, "repair_attempts": 0}

    result = constraint_validator_node(state)
    assert result["is_constraint_valid"] is False
    assert result["next_agent"] == "constraint_repair"
    assert len(result["constraint_errors"]) > 0


# =========================================================================
# 5. Milestone 5: Constraint Repair Node Tests (Story 3.2)
# =========================================================================
from src.agents.planner.repairs import (
    repair_budget_violations,
    repair_logistics_violations,
    constraint_repair_node,
    repair_fallback_node,
)
from src.agents.planner.graph import (
    create_planner_graph,
    planner_app,
    route_validation,
)
from src.agents.planner.temporal_validator import validate_temporal
from src.agents.planner.evaluator import plan_evaluation_fallback_node, plan_evaluation_node
from src.api.models import PlannerRequest
from src.api.planner import create_plan


def test_route_validation():
    """Verify all validation outcomes map to the expected graph branches."""
    assert route_validation({"is_constraint_valid": True, "repair_attempts": 0}) == "valid"
    assert route_validation({"is_constraint_valid": False, "repair_attempts": 2}) == "repair"
    assert route_validation({"is_constraint_valid": False, "repair_attempts": 3}) == "fallback"


def test_create_planner_graph_compiles():
    """Verify the Planner Part 1 graph factory returns a compiled graph."""
    assert create_planner_graph() is not None
    assert planner_app is not None


def test_planner_graph_valid_plan_ends_after_validation():
    """Verify a feasible draft reaches the Part 1 terminal boundary."""
    req = TripRequirements(destination="Cairo", duration_days=2, total_budget=500.0)
    result = planner_app.invoke({"trip_requirements": req, "repair_attempts": 0, "messages": []})

    assert result["is_constraint_valid"] is True
    assert result["current_plan"].status == "approved"
    assert result["is_advanced_valid"] is True
    assert result["evaluation"]["grade"] == "Good"


def test_planner_graph_falls_back_after_three_repairs():
    """Verify an infeasible draft follows the repair cycle and then falls back."""
    req = TripRequirements(destination="Cairo", duration_days=3, total_budget=20.0)
    result = planner_app.invoke({"trip_requirements": req, "repair_attempts": 0, "messages": []})

    assert result["is_constraint_valid"] is False
    assert result["repair_attempts"] == 3
    assert result["next_agent"] == "repair_fallback_completed"


def test_temporal_validator_reports_missing_metadata():
    """Verify strict temporal validation rejects incomplete activity metadata."""
    plan = GeneratedPlan(
        plan_id="p_temporal",
        days=[ItineraryDay(
            day_number=1,
            activities=[{"name": "Museum", "time_slot": "morning"}],
        )],
    )
    is_valid, errors = validate_temporal(plan)

    assert is_valid is False
    assert any("missing start_time" in error for error in errors)
    assert any("missing opening_time" in error for error in errors)


def test_planner_api_returns_plan_diagnostics():
    """Verify the Planner API returns an evaluated plan and diagnostics."""
    response = create_plan(
        PlannerRequest(destination="Cairo", duration_days=2, total_budget=500.0)
    )

    assert response.plan["status"] == "approved"
    assert response.is_constraint_valid is True
    assert response.is_advanced_valid is True
    assert response.evaluation["grade"] == "Good"


def test_poor_plan_preserves_best_effort_status_and_warning():
    """Verify Poor evaluation keeps a reviewable but non-executable plan."""
    plan = GeneratedPlan(plan_id="poor_1", title="Needs Review")
    evaluated = plan_evaluation_node({
        "current_plan": plan,
        "advanced_errors": ["Unresolved conflict", "Missing access"],
        "messages": [],
    })
    result = plan_evaluation_fallback_node(evaluated)

    assert result["plan_quality"] == "Poor"
    assert result["is_executable"] is False
    assert result["current_plan"].status == "review"
    assert "best-effort" in result["response"]


def test_repair_budget_violations():
    """Verify budget repair swaps expensive activities to reduce total cost."""
    activities = [
        {"name": "Luxury Tour", "cost_usd": 80.0, "transit_cost_usd": 10.0, "time_slot": "morning"},
        {"name": "Fine Dining", "cost_usd": 50.0, "transit_cost_usd": 5.0, "time_slot": "evening"},
    ]
    plan = GeneratedPlan(plan_id="p_rep_1", days=[ItineraryDay(day_number=1, activities=activities, estimated_cost=145.0)])
    repaired_plan = repair_budget_violations(plan, target_budget=50.0, overage=95.0)

    total_repaired_cost = sum(d.estimated_cost for d in repaired_plan.days)
    assert total_repaired_cost < 145.0
    swapped_names = [a["name"] for a in repaired_plan.days[0].activities]
    assert any("Koshary" in name or "Walking" in name or "Coptic" in name for name in swapped_names)


def test_repair_logistics_violations():
    """Verify logistics repair caps excessive transit times."""
    activities = [
        {"name": "Place 1", "travel_from_prev_mins": 120, "cost_usd": 10.0, "transit_cost_usd": 15.0},
    ]
    plan = GeneratedPlan(plan_id="p_rep_2", days=[ItineraryDay(day_number=1, activities=activities, estimated_cost=25.0)])
    repaired_plan = repair_logistics_violations(plan)

    assert repaired_plan.days[0].activities[0]["travel_from_prev_mins"] == 30
    assert repaired_plan.days[0].activities[0]["transit_cost_usd"] == 5.0


def test_constraint_repair_node():
    """Verify constraint repair node increments repair_attempts and routes to validator."""
    req = TripRequirements(destination="Cairo", total_budget=50.0, duration_days=2)
    activities = [
        {"name": "Expensive Tour", "cost_usd": 100.0, "transit_cost_usd": 10.0, "time_slot": "morning"}
    ]
    plan = GeneratedPlan(plan_id="p_rep_3", days=[ItineraryDay(day_number=1, activities=activities, estimated_cost=110.0)])
    state = {
        "current_plan": plan,
        "trip_requirements": req,
        "repair_attempts": 0,
        "constraint_errors": ["Total estimated cost $110.00 exceeds user budget of $50.00 by $60.00 USD."],
        "budget_overage": 60.0,
        "messages": []
    }

    result = constraint_repair_node(state)
    assert result["repair_attempts"] == 1
    assert result["next_agent"] == "constraint_validator"
    assert result["total_cost"] < 110.0
    assert "Constraint Repair (Attempt 1/3)" in result["response"]


def test_repair_fallback_node():
    """Verify repair fallback node provides diagnostic explanation when attempts exhaust."""
    state = {
        "constraint_errors": ["Budget exceeded by $100.00 USD"],
        "messages": []
    }
    result = repair_fallback_node(state)
    assert result["is_constraint_valid"] is False
    assert result["next_agent"] == "repair_fallback_completed"
    assert "Itinerary Constraint Notice" in result["response"]
    assert "Budget exceeded by $100.00 USD" in result["response"]




