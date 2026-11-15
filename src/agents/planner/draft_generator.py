"""
Planner Draft Generator Node (Story 3.1)
Synthesizes base tools (places search, travel times, weather forecast) into structured
candidate itineraries conforming to GeneratedPlan and ItineraryDay schemas.
Module: src/agents/planner/draft_generator.py
"""

from typing import Dict, Any, Optional, List
import uuid
from langchain_core.messages import AIMessage

from src.state.schema import GeneratedPlan, ItineraryDay, TripRequirements, UserPreferences
from src.tools.maps import search_local_places, calculate_travel_time
from src.tools.weather import get_weather_forecast


def build_draft_itinerary(
    requirements: TripRequirements,
    preferences: Optional[UserPreferences] = None
) -> GeneratedPlan:
    """
    Synthesizes tools outputs into a day-by-day candidate itinerary.
    """
    destination = requirements.destination or "Cairo"
    days_count = getattr(requirements, "duration_days", None) or 3
    if days_count < 1:
        days_count = 1
    elif days_count > 14:
        days_count = 14

    # 1. Fetch destination weather insights
    weather = get_weather_forecast(destination)

    # 2. Fetch candidate places and dining
    places = search_local_places(destination=destination, limit=20)
    attractions = [p for p in places if p.get("category") != "dining"]
    restaurants = [p for p in places if p.get("category") == "dining"]

    # Ensure fallback dining if none found in specific city
    if not restaurants:
        restaurants = search_local_places(destination="Cairo", category="dining")

    days: List[ItineraryDay] = []
    attr_idx = 0
    rest_idx = 0

    time_slots = ["morning", "afternoon", "evening"]

    for day_num in range(1, days_count + 1):
        day_activities: List[Dict[str, Any]] = []
        day_cost = 0.0
        prev_place_name = f"{destination} Hotel"

        # Morning Activity (usually prominent attraction)
        if attractions:
            morning_attr = attractions[attr_idx % len(attractions)]
            attr_idx += 1
            travel = calculate_travel_time(prev_place_name, morning_attr["name"])
            
            act_cost = morning_attr.get("estimated_cost_usd", 15.0)
            transit_cost = travel.get("estimated_transit_cost_usd", 5.0)
            day_cost += (act_cost + transit_cost)

            day_activities.append({
                "time_slot": "morning",
                "place_id": morning_attr.get("id"),
                "name": morning_attr["name"],
                "category": morning_attr.get("category", "sightseeing"),
                "cost_usd": act_cost,
                "duration_hours": morning_attr.get("recommended_duration_hours", 2.5),
                "travel_from_prev_mins": travel.get("duration_minutes", 20),
                "transit_cost_usd": transit_cost,
                "description": morning_attr.get("description", "")
            })
            prev_place_name = morning_attr["name"]

        # Afternoon Activity (secondary attraction or museum)
        if attractions:
            afternoon_attr = attractions[attr_idx % len(attractions)]
            attr_idx += 1
            travel = calculate_travel_time(prev_place_name, afternoon_attr["name"])
            
            act_cost = afternoon_attr.get("estimated_cost_usd", 15.0)
            transit_cost = travel.get("estimated_transit_cost_usd", 4.0)
            day_cost += (act_cost + transit_cost)

            day_activities.append({
                "time_slot": "afternoon",
                "place_id": afternoon_attr.get("id"),
                "name": afternoon_attr["name"],
                "category": afternoon_attr.get("category", "sightseeing"),
                "cost_usd": act_cost,
                "duration_hours": afternoon_attr.get("recommended_duration_hours", 2.0),
                "travel_from_prev_mins": travel.get("duration_minutes", 20),
                "transit_cost_usd": transit_cost,
                "description": afternoon_attr.get("description", "")
            })
            prev_place_name = afternoon_attr["name"]

        # Evening Activity (authentic dining or cultural stroll)
        if restaurants:
            dining = restaurants[rest_idx % len(restaurants)]
            rest_idx += 1
            travel = calculate_travel_time(prev_place_name, dining["name"])
            
            act_cost = dining.get("estimated_cost_usd", 10.0)
            transit_cost = travel.get("estimated_transit_cost_usd", 4.0)
            day_cost += (act_cost + transit_cost)

            day_activities.append({
                "time_slot": "evening",
                "place_id": dining.get("id"),
                "name": dining["name"],
                "category": "dining",
                "cost_usd": act_cost,
                "duration_hours": dining.get("recommended_duration_hours", 1.5),
                "travel_from_prev_mins": travel.get("duration_minutes", 15),
                "transit_cost_usd": transit_cost,
                "description": dining.get("description", "")
            })

        days.append(ItineraryDay(
            day_number=day_num,
            activities=day_activities,
            estimated_cost=round(day_cost, 2)
        ))

    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    title = f"{days_count}-Day Authentic Journey in {destination}"

    return GeneratedPlan(
        plan_id=plan_id,
        title=title,
        days=days,
        status="draft"
    )


def draft_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Story 3.1:
    Builds the draft itinerary and stores it into `current_plan`.
    """
    requirements = state.get("trip_requirements")
    if not isinstance(requirements, TripRequirements):
        requirements = TripRequirements(destination="Cairo", duration_days=3, total_budget=350.0)

    preferences = state.get("user_preferences")
    plan = build_draft_itinerary(requirements, preferences)

    # Format human-readable summary
    total_est_cost = sum(d.estimated_cost for d in plan.days)
    summary_lines = [
        f"📋 **Draft Itinerary: {plan.title}**",
        f"• Total Days: {len(plan.days)}",
        f"• Total Estimated Cost: ${total_est_cost:.2f} USD\n"
    ]
    for d in plan.days:
        summary_lines.append(f"**Day {d.day_number}** (Est: ${d.estimated_cost:.2f}):")
        for act in d.activities:
            summary_lines.append(f"  - [{act['time_slot'].title()}] {act['name']} (${act['cost_usd']:.2f})")
    
    summary_text = "\n".join(summary_lines)
    ai_msg = AIMessage(content=summary_text)

    return {
        **state,
        "current_plan": plan,
        "response": summary_text,
        "messages": state.get("messages", []) + [ai_msg],
        "next_agent": "constraint_validator"
    }
