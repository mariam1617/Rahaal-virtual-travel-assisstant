"""
Tools Package for Agents
"""

from .maps import search_local_places, calculate_travel_time, EGYPT_PLACES_DATABASE
from .weather import get_weather_forecast
from .booking_mock import (
    MockBookingAPI,
    search_activities,
    search_booking,
    search_flights,
    search_hotels,
)
from .calendar import create_calendar_event, sync_google_calendar_event

__all__ = [
    "search_local_places",
    "calculate_travel_time",
    "get_weather_forecast",
    "EGYPT_PLACES_DATABASE",
    "MockBookingAPI",
    "search_booking",
    "search_flights",
    "search_hotels",
    "search_activities",
    "create_calendar_event",
    "sync_google_calendar_event",
]
