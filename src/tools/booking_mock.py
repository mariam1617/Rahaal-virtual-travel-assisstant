"""Deterministic mock booking API for flights, hotels, and activities."""

from typing import Any, Dict, Optional
import re


VALID_BOOKING_TYPES = {"flight", "hotel", "activity"}


def search_booking(
    booking_type: str,
    destination: str,
    date: Optional[str] = None,
    query: Optional[str] = None,
    force_failure: bool = False,
) -> Dict[str, Any]:
    """Search or reserve an item through the mock booking endpoint."""
    normalized_type = str(booking_type).strip().lower()
    if normalized_type not in VALID_BOOKING_TYPES:
        return {
            "success": False,
            "status": "failed",
            "error": f"Unsupported booking type: {booking_type}",
            "booking_type": booking_type,
            "retryable": False,
        }
    if force_failure:
        return {
            "success": False,
            "status": "failed",
            "error": "Mock booking service temporarily unavailable",
            "booking_type": normalized_type,
            "retryable": True,
        }

    safe_destination = destination or "Unknown"
    item_name = (query or f"{normalized_type.title()} in {safe_destination}").strip()
    identifier = re.sub(r"[^a-z0-9]+", "-", item_name.lower()).strip("-")[:40]
    return {
        "success": True,
        "status": "confirmed",
        "booking_id": f"mock-{normalized_type}-{identifier or 'booking'}",
        "booking_type": normalized_type,
        "name": item_name,
        "destination": safe_destination,
        "date": date,
    }


class MockBookingAPI:
    """Small service wrapper for the deterministic mock endpoint."""

    def search(
        self,
        booking_type: str,
        destination: str,
        date: Optional[str] = None,
        query: Optional[str] = None,
        force_failure: bool = False,
    ) -> Dict[str, Any]:
        return search_booking(
            booking_type=booking_type,
            destination=destination,
            date=date,
            query=query,
            force_failure=force_failure,
        )


def search_flights(destination: str, date: Optional[str] = None, query: Optional[str] = None, force_failure: bool = False) -> Dict[str, Any]:
    return search_booking("flight", destination, date, query, force_failure)


def search_hotels(destination: str, date: Optional[str] = None, query: Optional[str] = None, force_failure: bool = False) -> Dict[str, Any]:
    return search_booking("hotel", destination, date, query, force_failure)


def search_activities(destination: str, date: Optional[str] = None, query: Optional[str] = None, force_failure: bool = False) -> Dict[str, Any]:
    return search_booking("activity", destination, date, query, force_failure)