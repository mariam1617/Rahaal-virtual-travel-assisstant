"""
Planner Constraint & Budget Validator Node (Story 3.2)
Calculates total costs and validates itinerary route feasibility against user budgets
and daily transit limitations.
Module: src/agents/planner/constraint_validator.py
"""

from typing import Dict, Any, Tuple, Optional, List
from src.state.schema import GeneratedPlan, TripRequirements

MAX_DAILY_TRANSIT_HOURS = 3.5


def validate_budget(
    plan: GeneratedPlan,
    max_budget: float
) -> Tuple[bool, Optional[str], float, float]:
    """
    Validates whether the total estimated cost of all days in the plan fits the budget.
    Returns: (is_valid, error_message, total_cost, overage)
    """
    if not plan or not plan.days:
        return True, None, 0.0, 0.0

    total_cost = sum(d.estimated_cost for d in plan.days)
    total_cost = round(total_cost, 2)

    if max_budget is not None and max_budget > 0 and total_cost > max_budget:
        overage = round(total_cost - max_budget, 2)
        err = f"Total estimated cost ${total_cost:.2f} exceeds user budget of ${max_budget:.2f} by ${overage:.2f} USD."
        return False, err, total_cost, overage

    return True, None, total_cost, 0.0


def validate_logistics(
    plan: GeneratedPlan,
    max_transit_hours: float = MAX_DAILY_TRANSIT_HOURS
) -> Tuple[bool, List[str]]:
    """
    Validates daily transit logistics to prevent unrealistic or physically exhausting travel.
    """
    if not plan or not plan.days:
        return True, []

    errors: List[str] = []
    max_mins = max_transit_hours * 60

    for day in plan.days:
        daily_transit_mins = sum(act.get("travel_from_prev_mins", 0) for act in day.activities)
        if daily_transit_mins > max_mins:
            hours = daily_transit_mins / 60.0
            errors.append(
                f"Day {day.day_number} transit time ({hours:.1f} hrs) exceeds comfortable maximum ({max_transit_hours} hrs)."
            )

    return len(errors) == 0, errors


def constraint_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Story 3.2:
    Validates the generated plan against budget and logistical constraints.
    Routes to `constraint_repair` if violations exist and attempts < 3.
    """
    plan = state.get("current_plan")
    if not isinstance(plan, GeneratedPlan):
        return {
            **state,
            "is_constraint_valid": False,
            "constraint_errors": ["No active plan available for constraint validation."],
            "next_agent": "draft_generator"
        }

    # Extract user budget
    requirements = state.get("trip_requirements")
    max_budget = None
    if isinstance(requirements, TripRequirements) and requirements.total_budget:
        max_budget = requirements.total_budget
    elif isinstance(requirements, dict) and requirements.get("total_budget"):
        max_budget = float(requirements["total_budget"])

    # 1. Budget validation
    budget_valid, budget_err, total_cost, overage = validate_budget(plan, max_budget or 99999.0)

    # 2. Logistics validation
    logistics_valid, logistics_errs = validate_logistics(plan)

    all_errors: List[str] = []
    if not budget_valid and budget_err:
        all_errors.append(budget_err)
    if not logistics_valid:
        all_errors.extend(logistics_errs)

    is_valid = budget_valid and logistics_valid
    attempts = state.get("repair_attempts", 0)

    if is_valid:
        plan.status = "validated"
        return {
            **state,
            "current_plan": plan,
            "is_constraint_valid": True,
            "constraint_errors": [],
            "total_cost": total_cost,
            "next_agent": "plan_validated"
        }

    # If invalid, prepare for repair loop or fallback cutoff
    next_node = "constraint_repair" if attempts < 3 else "repair_fallback"

    return {
        **state,
        "current_plan": plan,
        "is_constraint_valid": False,
        "constraint_errors": all_errors,
        "total_cost": total_cost,
        "budget_overage": overage,
        "next_agent": next_node
    }
