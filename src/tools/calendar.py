"""Google Calendar-compatible mock integration."""

from typing import Any, Dict, Optional
import re


def create_calendar_event(
    summary: str,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a Google Calendar-compatible event payload."""
    if not summary or not str(summary).strip():
        return {
            "success": False,
            "status": "failed",
            "error": "Calendar event summary is required.",
        }
    if not start_date:
        return {
            "success": False,
            "status": "failed",
            "error": "Calendar event start date is required.",
        }

    event_id = re.sub(r"[^a-z0-9]+", "-", str(summary).lower()).strip("-")[:40]
    start = (
        {"date": start_date}
        if not start_time
        else {"dateTime": f"{start_date}T{start_time}"}
    )
    end = (
        {"date": start_date}
        if not end_time
        else {"dateTime": f"{start_date}T{end_time}"}
    )
    return {
        "success": True,
        "status": "confirmed",
        "event_id": f"google-{event_id or 'event'}",
        "summary": str(summary).strip(),
        "location": location,
        "start": start,
        "end": end,
        "description": description or "Created by Rahaal travel assistant.",
    }


def sync_google_calendar_event(
    summary: str,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    force_failure: bool = False,
) -> Dict[str, Any]:
    """Push one event to the mock Google Calendar API."""
    if force_failure:
        return {
            "success": False,
            "status": "failed",
            "error": "Google Calendar service temporarily unavailable.",
        }
    return create_calendar_event(
        summary=summary,
        location=location,
        start_date=start_date,
        start_time=start_time,
        end_time=end_time,
        description=description,
    )