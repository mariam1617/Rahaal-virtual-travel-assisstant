"""
Planner Weather Tool - Weather Forecast & Climate Advisories (Story 3.1)
Module: src/tools/weather.py
"""

from typing import Dict, Any, Optional


DESTINATION_CLIMATES = {
    "cairo": {
        "city": "Cairo",
        "temp_high_c": 28,
        "temp_low_c": 17,
        "condition": "Sunny & Mild",
        "rain_probability_pct": 2,
        "heat_advisory": False,
        "advice": "Pleasant conditions for sightseeing. Pack a light jacket for breezy evenings by the Nile."
    },
    "giza": {
        "city": "Giza",
        "temp_high_c": 29,
        "temp_low_c": 17,
        "condition": "Sunny",
        "rain_probability_pct": 0,
        "heat_advisory": False,
        "advice": "Open desert sun at the Pyramids. Sunscreen, sunglasses, and comfortable walking footwear recommended."
    },
    "luxor": {
        "city": "Luxor",
        "temp_high_c": 34,
        "temp_low_c": 20,
        "condition": "Clear & Warm",
        "rain_probability_pct": 0,
        "heat_advisory": True,
        "advice": "Intense midday heat. Schedule Valley of the Kings and temple explorations early morning (07:00-11:00)."
    },
    "aswan": {
        "city": "Aswan",
        "temp_high_c": 35,
        "temp_low_c": 21,
        "condition": "Sunny & Dry",
        "rain_probability_pct": 0,
        "heat_advisory": True,
        "advice": "High daytime temperatures. Stay well-hydrated and plan Nile boating in late afternoon."
    },
    "alexandria": {
        "city": "Alexandria",
        "temp_high_c": 24,
        "temp_low_c": 16,
        "condition": "Breezy & Partly Cloudy",
        "rain_probability_pct": 12,
        "heat_advisory": False,
        "advice": "Fresh Mediterranean breeze. Great walking weather along the Corniche."
    },
    "hurghada": {
        "city": "Hurghada",
        "temp_high_c": 30,
        "temp_low_c": 22,
        "condition": "Sunny & Clear",
        "rain_probability_pct": 0,
        "heat_advisory": False,
        "advice": "Ideal conditions for Red Sea diving, snorkeling, and beach activities."
    }
}


def get_weather_forecast(destination: str = "Cairo", date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches the weather forecast and practical climate advisories for a given Egyptian destination.
    """
    dest_key = destination.lower().strip()
    
    # Check known destinations
    for key, data in DESTINATION_CLIMATES.items():
        if key in dest_key or dest_key in key:
            return {
                "destination": data["city"],
                "date": date or "upcoming",
                "temperature_high_c": data["temp_high_c"],
                "temperature_low_c": data["temp_low_c"],
                "condition": data["condition"],
                "rain_probability_pct": data["rain_probability_pct"],
                "heat_advisory": data["heat_advisory"],
                "advice": data["advice"]
            }

    # Default Egyptian climate fallback
    return {
        "destination": destination.title(),
        "date": date or "upcoming",
        "temperature_high_c": 28,
        "temperature_low_c": 18,
        "condition": "Sunny & Clear",
        "rain_probability_pct": 0,
        "heat_advisory": False,
        "advice": "Favorable travel conditions across Egypt. Stay hydrated during outdoor sightseeing."
    }
