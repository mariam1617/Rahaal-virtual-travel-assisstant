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
from src.agents.execution.calendar_sync import calendar_sync_node
from src.tools.booking_mock import (
    search_activities,
    search_booking,
    search_flights,
    search_hotels,
)
from src.tools.calendar import sync_google_calendar_event


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


def test_calendar_tool_creates_event_with_location_and_date():
    event = sync_google_calendar_event(
        summary="Grand Egyptian Museum",
        location="Cairo",
        start_date="2026-12-01",
        start_time="09:00",
        end_time="12:00",
        description="Confirmed booking ID: mock-activity-gem",
    )

    assert event["success"] is True
    assert event["status"] == "confirmed"
    assert event["location"] == "Cairo"
    assert event["start"]["dateTime"] == "2026-12-01T09:00"
    assert "booking ID" in event["description"]


def test_calendar_sync_requires_confirmed_booking():
    result = calendar_sync_node({
        "booking_status": "failed",
        "bookings": [],
    })

    assert result["calendar_status"] == "blocked"
    assert "confirmed bookings" in result["calendar_errors"][0]


def test_calendar_sync_writes_confirmed_booking():
    result = calendar_sync_node({
        "booking_status": "confirmed",
        "trip_requirements": TripRequirements(destination="Cairo"),
        "bookings": [{
            "success": True,
            "status": "confirmed",
            "name": "Grand Egyptian Museum",
            "destination": "Cairo",
            "date": "2026-12-01",
            "booking_id": "mock-activity-gem",
        }],
    })

    assert result["calendar_status"] == "synced"
    assert len(result["calendar_events"]) == 1
    assert result["next_agent"] == "end_execution"


def test_calendar_sync_reports_service_failure():
    result = calendar_sync_node({
        "booking_status": "confirmed",
        "mock_calendar_failure": True,
        "bookings": [{
            "status": "confirmed",
            "name": "Hotel",
            "destination": "Cairo",
            "date": "2026-12-01",
            "booking_id": "mock-hotel-cairo",
        }],
    })

    assert result["calendar_status"] == "failed"
    assert len(result["calendar_errors"]) == 1


def test_confirmed_execution_graph_syncs_calendar():
    plan = GeneratedPlan(
        plan_id="calendar-e2e",
        days=[ItineraryDay(
            day_number=1,
            activities=[{
                "name": "Grand Egyptian Museum",
                "booking_type": "activity",
                "date": "2026-12-01",
            }],
        )],
    )
    result = execution_app.invoke(
        {
            "approval_status": "approved",
            "booking_attempts": 0,
            "current_plan": plan,
            "trip_requirements": TripRequirements(destination="Cairo"),
        },
        config={"configurable": {"thread_id": "calendar-sync-e2e"}},
    )

    assert result["booking_status"] == "confirmed"
    assert result["calendar_status"] == "synced"
    assert result["next_agent"] == "end_execution"