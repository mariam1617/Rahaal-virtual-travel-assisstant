import pytest

from src.agents.execution.booking import (
    booking_fallback_node,
    execute_booking_with_retry,
    MAX_BOOKING_RETRIES,
    route_booking,
    search_booking_node,
)
from src.agents.execution.graph import create_execution_graph, execution_app
from src.state.schema import GeneratedPlan, ItineraryDay, TripRequirements
from src.tools.booking_mock import (
    search_activities,
    search_booking,
    search_flights,
    search_hotels,
)


def test_mock_booking_supports_all_resource_types():
    flight = search_flights("Cairo", date="2026-12-01")
    hotel = search_hotels("Cairo", date="2026-12-01")
    activity = search_activities("Cairo", date="2026-12-01")

    assert flight["status"] == "confirmed"
    assert hotel["status"] == "confirmed"
    assert activity["status"] == "confirmed"
    assert flight["booking_id"].startswith("mock-flight-")


def test_mock_booking_rejects_unknown_type():
    result = search_booking("train", "Cairo")
    assert result["success"] is False
    assert "Unsupported" in result["error"]


def test_booking_node_blocks_without_approval():
    plan = GeneratedPlan(
        plan_id="blocked",
        days=[ItineraryDay(day_number=1, activities=[{"name": "Museum"}])],
    )
    result = search_booking_node({
        "approval_status": "pending",
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo"),
    })

    assert result["booking_status"] == "blocked"
    assert "approval" in result["booking_errors"][0].lower()


def test_booking_node_confirms_after_approval():
    plan = GeneratedPlan(
        plan_id="approved",
        days=[ItineraryDay(
            day_number=1,
            activities=[{"name": "Grand Egyptian Museum", "booking_type": "activity"}],
        )],
    )
    result = search_booking_node({
        "approval_status": "approved",
        "booking_attempts": 0,
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo"),
    })

    assert result["booking_status"] == "confirmed"
    assert len(result["bookings"]) == 1
    assert result["next_agent"] == "booking_confirmed"


def test_booking_retries_three_times_then_fallback():
    plan = GeneratedPlan(
        plan_id="retry",
        days=[ItineraryDay(day_number=1, activities=[{"name": "Museum"}])],
    )
    state = {
        "approval_status": "approved",
        "booking_attempts": 0,
        "mock_booking_failure": True,
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo"),
    }
    result = execution_app.invoke(state, config={"configurable": {"thread_id": "booking-failure-test"}})

    assert result["booking_attempts"] == 3
    assert result["booking_status"] == "failed"
    assert result["next_agent"] == "booking_failed"


def test_booking_route_and_fallback_notification():
    assert route_booking({"booking_status": "confirmed", "booking_attempts": 1}) == "confirmed"
    assert route_booking({"booking_status": "failed", "booking_attempts": 2}) == "retry"
    assert route_booking({"booking_status": "failed", "booking_attempts": 3}) == "fallback"

    result = booking_fallback_node({"booking_attempts": 3, "booking_errors": ["unavailable"]})
    assert "3 attempts" in result["response"]


def test_execute_booking_with_retry_stops_at_success():
    attempts = {"count": 0}

    def transient_booking():
        attempts["count"] += 1
        if attempts["count"] < 3:
            return {"success": False, "retryable": True, "error": "temporary"}
        return {"success": True, "status": "confirmed", "booking_id": "mock-recovered"}

    result = execute_booking_with_retry(transient_booking)

    assert result["success"] is True
    assert attempts["count"] == 3


def test_execute_booking_with_retry_hard_limit():
    result = execute_booking_with_retry(
        lambda: {"success": False, "retryable": True, "error": "offline"}
    )

    assert result["attempts"] == MAX_BOOKING_RETRIES
    assert result["success"] is False


def test_execute_booking_with_retry_does_not_retry_permanent_error():
    attempts = 0

    def invalid_booking():
        nonlocal attempts
        attempts += 1
        return {"success": False, "retryable": False, "error": "Unsupported booking type"}

    result = execute_booking_with_retry(invalid_booking)

    assert result["retryable"] is False
    assert attempts == 1


def test_booking_node_recovers_after_transient_failures():
    plan = GeneratedPlan(
        plan_id="recover",
        days=[ItineraryDay(day_number=1, activities=[{"name": "Museum"}])],
    )
    result = search_booking_node({
        "approval_status": "approved",
        "mock_booking_failures_remaining": 2,
        "booking_attempts": 0,
        "current_plan": plan,
        "trip_requirements": TripRequirements(destination="Cairo"),
    })

    assert result["booking_status"] == "confirmed"
    assert result["booking_attempts"] == 3


def test_execution_graph_compiles():
    assert create_execution_graph() is not None
    assert execution_app is not None