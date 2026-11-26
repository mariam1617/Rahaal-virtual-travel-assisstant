"""
Planner Geographic Tools - Places Lookup & Travel Time Calculation (Story 3.1)
Module: src/tools/maps.py
"""

from typing import List, Dict, Any, Optional
import math


# Pre-compiled catalog of major Egyptian tourist hubs, attractions, and dining
EGYPT_PLACES_DATABASE: List[Dict[str, Any]] = [
    # --- Cairo & Giza ---
    {
        "id": "cai_pyramids",
        "name": "Pyramids of Giza & Great Sphinx",
        "destination": "Cairo",
        "city": "Giza",
        "category": "historical",
        "estimated_cost_usd": 20.0,
        "recommended_duration_hours": 3.5,
        "coordinates": {"lat": 29.9792, "lng": 31.1342},
        "description": "The iconic ancient wonders of the Old Kingdom, including Khufu, Khafre, Menkaure, and the Sphinx."
    },
    {
        "id": "cai_gem",
        "name": "The Grand Egyptian Museum (GEM)",
        "destination": "Cairo",
        "city": "Giza",
        "category": "museum",
        "estimated_cost_usd": 25.0,
        "recommended_duration_hours": 3.0,
        "coordinates": {"lat": 29.9950, "lng": 31.1198},
        "description": "The world's largest archaeological museum complex dedicated to ancient Egyptian civilization."
    },
    {
        "id": "cai_egyptian_museum",
        "name": "The Egyptian Museum in Tahrir",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "museum",
        "estimated_cost_usd": 15.0,
        "recommended_duration_hours": 2.5,
        "coordinates": {"lat": 30.0478, "lng": 31.2336},
        "description": "Historic museum housing over 120,000 antiquities in downtown Cairo."
    },
    {
        "id": "cai_khan_elkhalili",
        "name": "Khan el-Khalili & Al-Mu'izz Street",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "cultural",
        "estimated_cost_usd": 0.0,
        "recommended_duration_hours": 2.5,
        "coordinates": {"lat": 30.0477, "lng": 31.2622},
        "description": "Historic Islamic Cairo bazaar with spice merchants, artisanal crafts, and Mamluk architecture."
    },
    {
        "id": "cai_citadel",
        "name": "Cairo Citadel & Mosque of Muhammad Ali",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "historical",
        "estimated_cost_usd": 15.0,
        "recommended_duration_hours": 2.5,
        "coordinates": {"lat": 30.0299, "lng": 31.2599},
        "description": "Medieval Islamic fortress with panoramic views over Cairo and the Alabaster Mosque."
    },
    {
        "id": "cai_coptic",
        "name": "Coptic Cairo & The Hanging Church",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "historical",
        "estimated_cost_usd": 0.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 30.0053, "lng": 31.2301},
        "description": "Ancient historic quarter featuring Babylon Fortress, Church of St. Sergius, and the Hanging Church."
    },
    {
        "id": "cai_abou_tarek",
        "name": "Koshary Abou Tarek",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "dining",
        "estimated_cost_usd": 5.0,
        "recommended_duration_hours": 1.0,
        "coordinates": {"lat": 30.0526, "lng": 31.2372},
        "description": "Egypt's most renowned multi-story establishment for authentic Egyptian Koshary."
    },
    {
        "id": "cai_nile_felucca",
        "name": "Sunset Nile Felucca Ride",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "activity",
        "estimated_cost_usd": 20.0,
        "recommended_duration_hours": 1.5,
        "coordinates": {"lat": 30.0390, "lng": 31.2260},
        "description": "Traditional wooden sailboat experience gliding along the Nile at sunset in Zamalek/Garden City."
    },
    {
        "id": "cai_azhar_park",
        "name": "Al-Azhar Park Gardens & Hilltop Dining",
        "destination": "Cairo",
        "city": "Cairo",
        "category": "nature",
        "estimated_cost_usd": 8.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 30.0409, "lng": 31.2647},
        "description": "Lush hillside park offering open gardens and sunset vistas of Old Cairo's minarets."
    },

    # --- Luxor ---
    {
        "id": "lux_karnak",
        "name": "Karnak Temple Complex",
        "destination": "Luxor",
        "city": "Luxor",
        "category": "historical",
        "estimated_cost_usd": 18.0,
        "recommended_duration_hours": 3.0,
        "coordinates": {"lat": 25.7188, "lng": 32.6573},
        "description": "Massive complex of pylons, chapels, and the monumental Great Hypostyle Hall on the East Bank."
    },
    {
        "id": "lux_temple",
        "name": "Luxor Temple by Night",
        "destination": "Luxor",
        "city": "Luxor",
        "category": "historical",
        "estimated_cost_usd": 14.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 25.6995, "lng": 32.6394},
        "description": "Ancient temple illuminated beautifully after sunset in central Luxor on the Nile corniche."
    },
    {
        "id": "lux_valley_kings",
        "name": "Valley of the Kings",
        "destination": "Luxor",
        "city": "Luxor",
        "category": "historical",
        "estimated_cost_usd": 22.0,
        "recommended_duration_hours": 3.5,
        "coordinates": {"lat": 25.7402, "lng": 32.6014},
        "description": "The burial site of New Kingdom Pharaohs, including Tutankhamun and Ramses IV on the West Bank."
    },
    {
        "id": "lux_hatshepsut",
        "name": "Mortuary Temple of Hatshepsut",
        "destination": "Luxor",
        "city": "Luxor",
        "category": "historical",
        "estimated_cost_usd": 12.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 25.7382, "lng": 32.6065},
        "description": "Terraced colonnaded temple carved dramatically into the cliffs of Deir el-Bahari."
    },
    {
        "id": "lux_sofra",
        "name": "Sofra Restaurant & Cafe",
        "destination": "Luxor",
        "city": "Luxor",
        "category": "dining",
        "estimated_cost_usd": 16.0,
        "recommended_duration_hours": 1.5,
        "coordinates": {"lat": 25.6961, "lng": 32.6418},
        "description": "Authentic Upper Egyptian cuisine served in a charming vintage oriental townhouse."
    },

    # --- Aswan ---
    {
        "id": "asw_philae",
        "name": "Philae Temple (Temple of Isis)",
        "destination": "Aswan",
        "city": "Aswan",
        "category": "historical",
        "estimated_cost_usd": 18.0,
        "recommended_duration_hours": 2.5,
        "coordinates": {"lat": 24.0253, "lng": 32.8842},
        "description": "Magnificently preserved Ptolemaic temple complex dedicated to Isis, located on Agilkia Island."
    },
    {
        "id": "asw_high_dam",
        "name": "Aswan High Dam",
        "destination": "Aswan",
        "city": "Aswan",
        "category": "historical",
        "estimated_cost_usd": 5.0,
        "recommended_duration_hours": 1.5,
        "coordinates": {"lat": 23.9700, "lng": 32.8775},
        "description": "One of the world's largest dams, creating Lake Nasser and transforming Egypt's agriculture."
    },
    {
        "id": "asw_felucca",
        "name": "Nile Felucca Ride to Elephantine Island",
        "destination": "Aswan",
        "city": "Aswan",
        "category": "activity",
        "estimated_cost_usd": 12.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 24.0869, "lng": 32.8826},
        "description": "Scenic traditional sailboat ride between Aswan's islands, passing Nubian villages and ruins."
    },
    {
        "id": "asw_nubian_village",
        "name": "Nubian Village Cultural Tour",
        "destination": "Aswan",
        "city": "Aswan",
        "category": "cultural",
        "estimated_cost_usd": 10.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 24.1024, "lng": 32.8759},
        "description": "Colorful Nubian community on the west bank, with traditional crafts, henna art, and tea."
    },
    {
        "id": "asw_1902",
        "name": "1902 Restaurant at the Old Cataract Hotel",
        "destination": "Aswan",
        "city": "Aswan",
        "category": "dining",
        "estimated_cost_usd": 30.0,
        "recommended_duration_hours": 1.5,
        "coordinates": {"lat": 24.0784, "lng": 32.8905},
        "description": "Iconic fine-dining restaurant in the historic Old Cataract Hotel with sweeping Nile views."
    },

    # --- Alexandria ---
    {
        "id": "alx_bibliotheca",
        "name": "Bibliotheca Alexandrina",
        "destination": "Alexandria",
        "city": "Alexandria",
        "category": "cultural",
        "estimated_cost_usd": 10.0,
        "recommended_duration_hours": 2.5,
        "coordinates": {"lat": 31.2089, "lng": 29.9092},
        "description": "World-renowned modern cultural center and library commemorating the ancient Library of Alexandria."
    },
    {
        "id": "alx_qaitbay",
        "name": "Citadel of Qaitbay",
        "destination": "Alexandria",
        "city": "Alexandria",
        "category": "historical",
        "estimated_cost_usd": 8.0,
        "recommended_duration_hours": 2.0,
        "coordinates": {"lat": 31.2140, "lng": 29.8856},
        "description": "15th-century defensive fortress situated on the Mediterranean coast on the site of the ancient Pharos Lighthouse."
    },
    {
        "id": "alx_greek_club",
        "name": "White and Blue Greek Club Dining",
        "destination": "Alexandria",
        "city": "Alexandria",
        "category": "dining",
        "estimated_cost_usd": 24.0,
        "recommended_duration_hours": 1.5,
        "coordinates": {"lat": 31.2135, "lng": 29.8860},
        "description": "Fresh Mediterranean seafood dining with waterfront views overlooking the Eastern Harbor."
    }
]


def search_local_places(
    query: Optional[str] = None,
    destination: str = "Cairo",
    category: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Searches candidate places and activities in Egypt filtered by destination and category.
    """
    dest_lower = destination.lower().strip()
    results = []

    for place in EGYPT_PLACES_DATABASE:
        # Check destination match (e.g. "cairo", "giza", "luxor", "alexandria")
        place_dest = place["destination"].lower()
        place_city = place["city"].lower()
        
        matches_dest = (dest_lower in place_dest) or (dest_lower in place_city) or (place_dest in dest_lower)
        if not matches_dest and dest_lower not in ["egypt", "all"]:
            continue

        # Check category match if provided
        if category and place["category"].lower() != category.lower().strip():
            continue

        # Check query match if provided
        if query:
            q_lower = query.lower().strip()
            name_match = q_lower in place["name"].lower()
            desc_match = q_lower in place["description"].lower()
            if not (name_match or desc_match):
                continue

        results.append(place)

    # If destination has no specific matches, fallback to Cairo attractions
    if not results and dest_lower not in ["cairo", "giza"]:
        return search_local_places(query=query, destination="Cairo", category=category, limit=limit)

    return results[:limit]


def calculate_travel_time(
    origin: str,
    destination: str,
    mode: str = "driving"
) -> Dict[str, Any]:
    """
    Calculates estimated travel time and distance between key Egyptian locations.
    """
    orig_clean = origin.lower().strip()
    dest_clean = destination.lower().strip()

    # Pre-calculated benchmark travel times within major hubs
    known_routes = {
        ("downtown cairo", "giza pyramids"): {"duration_mins": 45, "distance_km": 18.5, "cost_usd": 8.0},
        ("giza pyramids", "grand egyptian museum"): {"duration_mins": 15, "distance_km": 4.5, "cost_usd": 4.0},
        ("downtown cairo", "khan el-khalili"): {"duration_mins": 25, "distance_km": 6.0, "cost_usd": 4.0},
        ("khan el-khalili", "cairo citadel"): {"duration_mins": 20, "distance_km": 5.2, "cost_usd": 3.5},
        ("cairo", "alexandria"): {"duration_mins": 160, "distance_km": 218.0, "cost_usd": 25.0},
        ("karnak temple", "luxor temple"): {"duration_mins": 15, "distance_km": 3.5, "cost_usd": 3.0},
        ("luxor east bank", "valley of the kings"): {"duration_mins": 35, "distance_km": 24.0, "cost_usd": 15.0},
        ("valley of the kings", "hatshepsut temple"): {"duration_mins": 10, "distance_km": 3.0, "cost_usd": 4.0},
    }

    # Direct match or reversed
    for (o, d), data in known_routes.items():
        if (o in orig_clean and d in dest_clean) or (d in orig_clean and o in dest_clean):
            return {
                "origin": origin,
                "destination": destination,
                "duration_minutes": data["duration_mins"],
                "distance_km": data["distance_km"],
                "estimated_transit_cost_usd": data["cost_usd"],
                "mode": mode
            }

    # Generic intra-city fallback
    return {
        "origin": origin,
        "destination": destination,
        "duration_minutes": 25,
        "distance_km": 8.0,
        "estimated_transit_cost_usd": 5.0,
        "mode": mode
    }
