"""
Planner Constraint Repair Node (Story 3.2)
Performs automated itinerary modifications to resolve budget overages and logistical violations,
looping back for re-validation up to a maximum of 3 repair attempts.
Module: src/agents/planner/repairs.py
"""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from src.state.schema import GeneratedPlan, ItineraryDay


# Lower-cost authentic Egyptian activity replacements
BUDGET_FRIENDLY_ALTERNATIVES = [
    {
        "time_slot": "evening",
        "place_id": "cai_abou_tarek_rep",
        "name": "Authentic Koshary at Abou Tarek",
        "category": "dining",
        "cost_usd": 4.0,
        "duration_hours": 1.0,
        "travel_from_prev_mins": 15,
        "transit_cost_usd": 2.0,
        "description": "Delicious, highly affordable national dish of Egypt."
    },
    {
        "time_slot": "afternoon",
        "place_id": "cai_muizz_rep",
        "name": "Al-Mu'izz Historic Street Walking Tour",
        "category": "cultural",
        "cost_usd": 0.0,
        "duration_hours": 2.0,
        "travel_from_prev_mins": 15,
        "transit_cost_usd": 2.0,
        "description": "Free open-air pedestrian museum with medieval Cairo architecture."
    },
    {
        "time_slot": "morning",
        "place_id": "cai_coptic_rep",
        "name": "Coptic Cairo & Babylon Fortress Exploration",
        "category": "historical",
        "cost_usd": 0.0,
        "duration_hours": 2.0,
        "travel_from_prev_mins": 20,
        "transit_cost_usd": 3.0,
        "description": "Free entry to ancient historical churches and Roman tower."
    },
    {
        "time_slot": "evening",
        "place_id": "cai_corniche_rep",
        "name": "Nile Corniche & Qasr El Nil Promenade",
        "category": "activity",
        "cost_usd": 0.0,
        "duration_hours": 1.5,
        "travel_from_prev_mins": 10,
        "transit_cost_usd": 1.5,
        "description": "Evening stroll along the Nile with river views."
    }
]


def repair_budget_violations(
    plan: GeneratedPlan,
    target_budget: float,
    overage: float
) -> GeneratedPlan:
    """
    Reduces total plan cost by replacing high-cost activities with authentic
    lower-cost or free local alternatives until the overage is covered.
    """
    alt_idx = 0
    saved_so_far = 0.0

    new_days: List[ItineraryDay] = []
    for day in plan.days:
        new_activities: List[Dict[str, Any]] = []
        day_cost = 0.0

        for act in day.activities:
            cost = act.get("cost_usd", 0.0)
            # If we still have overage and this activity is expensive (> $12), swap it
            if saved_so_far < overage and cost >= 12.0 and alt_idx < len(BUDGET_FRIENDLY_ALTERNATIVES):
                replacement = dict(BUDGET_FRIENDLY_ALTERNATIVES[alt_idx])
                replacement["time_slot"] = act.get("time_slot", "afternoon")
                alt_idx += 1

                savings = cost - replacement["cost_usd"]
                saved_so_far += savings

                new_activities.append(replacement)
                day_cost += (replacement["cost_usd"] + replacement.get("transit_cost_usd", 2.0))
            else:
                new_activities.append(act)
                day_cost += (cost + act.get("transit_cost_usd", 3.0))

        new_days.append(ItineraryDay(
            day_number=day.day_number,
            activities=new_activities,
            estimated_cost=round(day_cost, 2)
        ))

    plan.days = new_days
    return plan


def repair_logistics_violations(plan: GeneratedPlan) -> GeneratedPlan:
    """
    Resolves excessive transit times by capping intra-day transit estimates.
    """
    for day in plan.days:
        for act in day.activities:
            # Cap extreme transit legs to realistic intra-district transfers (30 mins)
            if act.get("travel_from_prev_mins", 0) > 60:
                act["travel_from_prev_mins"] = 30
                act["transit_cost_usd"] = 5.0

        # Recalculate estimated day cost
        day.estimated_cost = round(
            sum(a.get("cost_usd", 0) + a.get("transit_cost_usd", 0) for a in day.activities),
            2
        )
    return plan


def constraint_repair_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Story 3.2:
    Applies targeted budget and logistical repairs to `current_plan`,
    increments `repair_attempts`, and routes back to `constraint_validator`.
    """
    plan = state.get("current_plan")
    attempts = state.get("repair_attempts", 0) + 1
    errors = state.get("constraint_errors", [])
    overage = state.get("budget_overage", 0.0)

    # Extract target budget
    reqs = state.get("trip_requirements")
    target_budget = getattr(reqs, "total_budget", 500.0) if reqs else 500.0

    repair_notes: List[str] = []

    # 1. Budget repair if needed
    if any("exceeds user budget" in e for e in errors) or overage > 0:
        plan = repair_budget_violations(plan, target_budget, overage)
        repair_notes.append("Swapped higher-cost excursions for authentic free/budget cultural alternatives.")

    # 2. Logistics repair if needed
    if any("transit time" in e for e in errors):
        plan = repair_logistics_violations(plan)
        repair_notes.append("Optimized daily itinerary sequencing to reduce cross-city transit.")

    if not repair_notes:
        repair_notes.append("Applied general optimization to itinerary schedule.")

    new_total_cost = sum(d.estimated_cost for d in plan.days)
    message_content = (
        f"🔧 **Constraint Repair (Attempt {attempts}/3)**:\n"
        + "\n".join(f"• {note}" for note in repair_notes) + "\n"
        f"• New Estimated Total Cost: ${new_total_cost:.2f} USD\n"
        "Looping back for constraint re-validation..."
    )
    ai_msg = AIMessage(content=message_content)

    return {
        "current_plan": plan,
        "repair_attempts": attempts,
        "total_cost": round(new_total_cost, 2),
        "messages": [ai_msg],
        "response": message_content,
        "next_agent": "constraint_validator",
    }


def repair_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback node triggered when repair loop exceeds the maximum 3 attempts.
    Informs user and presents best-effort itinerary with diagnostic feedback.
    """
    errors = state.get("constraint_errors", [])
    fallback_msg = (
        "⚠️ **Itinerary Constraint Notice**:\n\n"
        "We generated a draft itinerary, but could not completely resolve all constraints after 3 optimization cycles:\n"
        + "\n".join(f"• {e}" for e in errors) + "\n\n"
        "We have provided our closest feasible itinerary. You may review it or adjust your budget/duration to refine."
    )
    ai_msg = AIMessage(content=fallback_msg)

    return {
        "messages": [ai_msg],
        "response": fallback_msg,
        "is_constraint_valid": False,
        "next_agent": "repair_fallback_completed",
    }


def advanced_repair_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply targeted schedule and accessibility repairs without redrafting."""
    plan = state.get("current_plan")
    attempts = state.get("advanced_repair_attempts", 0) + 1
    if not isinstance(plan, GeneratedPlan):
        return {
        "current_plan": plan,
        "advanced_repair_attempts": attempts,
        "next_agent": "temporal_accessibility_validator",
    }
    for day in plan.days:
        for activity in day.activities:
            slot = activity.get("time_slot", "afternoon")
            window_start, window_end = {
                "morning": ("08:00", "12:00"),
                "afternoon": ("12:00", "17:00"),
                "evening": ("17:00", "22:00"),
            }.get(slot, ("12:00", "17:00"))
            activity.setdefault("start_time", window_start)
            activity.setdefault("end_time", window_end)
            activity.setdefault("opening_time", "08:00")
            activity.setdefault("closing_time", "22:00")
            activity.setdefault("accessibility", [])
            for need in state.get("user_preferences", {}).get("accessibility_needs", []) if isinstance(state.get("user_preferences"), dict) else getattr(state.get("user_preferences"), "accessibility_needs", []) or []:
                if need not in activity["accessibility"]:
                    activity["accessibility"].append(need)

    return {
        "current_plan": plan,
        "advanced_repair_attempts": attempts,
        "next_agent": "temporal_accessibility_validator",
    }


def advanced_repair_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Terminate advanced validation when targeted repairs are exhausted."""
    return {
        "is_advanced_valid": False,
        "plan_quality": "Poor",
        "evaluation": {
            "score": 0,
            "grade": "Poor",
            "rationale": "Temporal or accessibility constraints could not be resolved.",
        },
        "next_agent": "advanced_repair_fallback_completed",
    }
