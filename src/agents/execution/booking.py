"""Booking nodes for the Execution Subgraph."""

from typing import Any, Callable, Dict

from src.state.schema import GeneratedPlan
from src.tools.booking_mock import search_booking


MAX_BOOKING_RETRIES = 3


def execute_booking_with_retry(
    booking_call: Callable[[], Dict[str, Any]],
    max_attempts: int = MAX_BOOKING_RETRIES,
) -> Dict[str, Any]:
    """Call the mock booking API up to the hard attempt limit."""
    last_result: Dict[str, Any] = {
        "success": False,
        "status": "failed",
        "error": "Booking attempt did not return a result.",
        "retryable": True,
    }

    for _attempt in range(1, max_attempts + 1):
        try:
            result = booking_call()
            if not isinstance(result, dict):
                raise TypeError("Booking API returned a non-object result.")
            last_result = result
            if result.get("success") is True:
                return result
            if result.get("retryable") is False:
                return result
        except Exception as exc:
            last_result = {
                "success": False,
                "status": "failed",
                "error": f"Booking service error: {exc}",
                "retryable": True,
            }

    last_result["attempts"] = max_attempts
    return last_result


def _booking_items(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    plan = state.get("current_plan")
    if not isinstance(plan, GeneratedPlan):
        return []

    items: list[Dict[str, Any]] = []
    for day in plan.days:
        for activity in day.activities:
            name = activity.get("name")
            if name:
                items.append({
                    "booking_type": activity.get("booking_type", "activity"),
                    "query": name,
                    "date": activity.get("date") or getattr(plan, "start_date", None),
                })
    return items


def search_booking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Book planned items only with approval, retrying transient failures."""
    if state.get("approval_status") != "approved":
        return {
            **state,
            "booking_status": "blocked",
            "booking_attempts": 0,
            "bookings": [],
            "booking_errors": ["Booking requires explicit itinerary approval."],
            "is_executable": False,
            "next_agent": "booking_blocked",
        }

    trip_requirements = state.get("trip_requirements")
    destination = getattr(trip_requirements, "destination", "Unknown")
    items = _booking_items(state)
    if not items:
        return {
            **state,
            "booking_status": "failed",
            "booking_attempts": int(state.get("booking_attempts", 0)) + 1,
            "bookings": [],
            "booking_errors": ["No bookable itinerary items were found."],
            "next_agent": "booking_retry",
        }

    bookings = []
    errors = []
    highest_attempts = 0
    force_failure = state.get("mock_booking_failure", False)
    failures_remaining = state.get("mock_booking_failures_remaining")

    for item in items:
        attempts_used = 0

        def call_booking() -> Dict[str, Any]:
            nonlocal attempts_used, failures_remaining
            attempts_used += 1
            if failures_remaining is not None:
                should_fail = int(failures_remaining) > 0
                failures_remaining = max(0, int(failures_remaining) - 1)
            else:
                should_fail = force_failure
            result = search_booking(
                booking_type=item["booking_type"],
                destination=destination,
                date=item["date"],
                query=item["query"],
                force_failure=should_fail,
            )
            if result["success"]:
                return result
            if "unsupported" in str(result.get("error", "")).lower():
                result["retryable"] = False
            else:
                result["retryable"] = True
            return result

        result = execute_booking_with_retry(call_booking)
        highest_attempts = max(highest_attempts, result.get("attempts", attempts_used))
        if result.get("success"):
            bookings.append(result)
        else:
            errors.append(result.get("error", "Booking failed"))

    attempts = max(int(state.get("booking_attempts", 0)), highest_attempts)
    if not errors:
        plan = state.get("current_plan")
        if isinstance(plan, GeneratedPlan):
            plan.status = "booked"
        return {
            **state,
            "current_plan": plan,
            "booking_status": "confirmed",
            "booking_attempts": attempts,
            "bookings": bookings,
            "booking_errors": [],
            "next_agent": "booking_confirmed",
        }

    return {
        **state,
        "booking_status": "failed",
        "booking_attempts": attempts,
        "bookings": bookings,
        "booking_errors": errors,
        "next_agent": "booking_retry",
    }


def booking_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Notify the user after the hard retry limit is reached."""
    response = (
        f"Booking could not be completed after {MAX_BOOKING_RETRIES} attempts. "
        "No further booking actions are being attempted."
    )
    return {
        **state,
        "booking_status": "failed",
        "booking_errors": state.get("booking_errors", []) or ["Booking retry limit reached."],
        "response": response,
        "next_agent": "booking_failed",
    }


def route_booking(state: Dict[str, Any]) -> str:
    """Route confirmed bookings to completion or retry/fallback."""
    if state.get("booking_status") == "confirmed":
        return "confirmed"
    if state.get("booking_status") == "blocked":
        return "blocked"
    if int(state.get("booking_attempts", 0)) < MAX_BOOKING_RETRIES:
        return "retry"
    return "fallback"


execute_booking_node = search_booking_node