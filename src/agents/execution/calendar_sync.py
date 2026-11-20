"""Calendar synchronization node for confirmed bookings."""

from typing import Any, Dict

from src.tools.calendar import sync_google_calendar_event


def calendar_sync_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Write confirmed bookings to Google Calendar only after booking success."""
    if state.get("booking_status") != "confirmed":
        return {
            **state,
            "calendar_status": "blocked",
            "calendar_events": [],
            "calendar_errors": ["Calendar sync requires confirmed bookings."],
            "next_agent": "calendar_sync_blocked",
        }

    bookings = state.get("bookings", [])
    trip_requirements = state.get("trip_requirements")
    destination = getattr(trip_requirements, "destination", None)
    force_failure = state.get("mock_calendar_failure", False)
    events = []
    errors = []

    for booking in bookings:
        if not isinstance(booking, dict) or booking.get("status") != "confirmed":
            continue
        result = sync_google_calendar_event(
            summary=booking.get("name") or f"{booking.get('booking_type', 'Booking').title()} booking",
            location=booking.get("destination") or destination,
            start_date=booking.get("date"),
            description=f"Confirmed booking ID: {booking.get('booking_id', 'unknown')}",
            force_failure=force_failure,
        )
        if result.get("success"):
            events.append(result)
        else:
            errors.append(result.get("error", "Calendar event could not be created."))

    if not bookings:
        return {
            **state,
            "calendar_status": "blocked",
            "calendar_events": [],
            "calendar_errors": ["No confirmed bookings are available to sync."],
            "next_agent": "calendar_sync_blocked",
        }
    if errors:
        return {
            **state,
            "calendar_status": "partial" if events else "failed",
            "calendar_events": events,
            "calendar_errors": errors,
            "next_agent": "end_execution",
        }

    return {
        **state,
        "calendar_status": "synced",
        "calendar_events": events,
        "calendar_errors": [],
        "next_agent": "end_execution",
    }