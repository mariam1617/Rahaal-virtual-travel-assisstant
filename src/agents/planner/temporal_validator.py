"""Temporal and accessibility validation for the Planner Subgraph."""

from typing import Any, Dict, List, Tuple

from src.state.schema import GeneratedPlan, UserPreferences


SLOT_WINDOWS = {
    "morning": (8, 12),
    "afternoon": (12, 17),
    "evening": (17, 22),
}


def _hour(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        parts = text.split(":")
        return float(parts[0]) + (float(parts[1]) / 60 if len(parts) > 1 else 0)
    except (TypeError, ValueError):
        return None


def validate_temporal(plan: GeneratedPlan) -> Tuple[bool, List[str]]:
    """Validate activity schedules against slot windows and opening hours."""
    errors: List[str] = []
    for day in plan.days:
        previous_end = None
        for activity in day.activities:
            slot = activity.get("time_slot")
            start = _hour(activity.get("start_time"))
            end = _hour(activity.get("end_time"))
            opening = _hour(activity.get("opening_time"))
            closing = _hour(activity.get("closing_time"))
            if start is None or end is None:
                errors.append(
                    f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' "
                    "is missing start_time or end_time."
                )
            elif end <= start:
                errors.append(f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' has an invalid time range.")
            if start is not None and end is not None and slot in SLOT_WINDOWS:
                window_start, window_end = SLOT_WINDOWS[slot]
                if start < window_start or end > window_end:
                    errors.append(
                        f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' "
                        f"falls outside its {slot} window."
                    )
            if opening is None or closing is None:
                errors.append(
                    f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' "
                    "is missing opening_time or closing_time."
                )
            elif start < opening or end > closing:
                errors.append(
                    f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' "
                    "is scheduled outside its opening hours."
                )
            if previous_end is not None and start is not None and start < previous_end:
                errors.append(f"Day {day.day_number} contains overlapping activities.")
            previous_end = end if end is not None else previous_end
    return not errors, errors


def validate_accessibility(
    plan: GeneratedPlan,
    preferences: UserPreferences | None,
) -> Tuple[bool, List[str]]:
    """Validate requested accessibility needs against activity capabilities."""
    needs = list(preferences.accessibility_needs or []) if preferences else []
    if not needs:
        return True, []

    errors: List[str] = []
    for day in plan.days:
        for activity in day.activities:
            capabilities = {
                str(value).lower()
                for value in activity.get("accessibility", [])
            }
            missing = [need for need in needs if str(need).lower() not in capabilities]
            if missing:
                errors.append(
                    f"Day {day.day_number} activity '{activity.get('name', 'unknown')}' "
                    f"does not support: {', '.join(missing)}."
                )
    return not errors, errors


def temporal_accessibility_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run strict temporal and accessibility checks on the current plan."""
    plan = state.get("current_plan")
    if not isinstance(plan, GeneratedPlan):
        errors = ["No active plan available for temporal/accessibility validation."]
        return {
            **state,
            "is_advanced_valid": False,
            "temporal_errors": errors,
            "accessibility_errors": [],
            "advanced_errors": errors,
            "next_agent": "advanced_repair",
        }

    temporal_valid, temporal_errors = validate_temporal(plan)
    accessibility_valid, accessibility_errors = validate_accessibility(
        plan, state.get("user_preferences")
    )
    errors = temporal_errors + accessibility_errors
    return {
        **state,
        "is_temporal_valid": temporal_valid,
        "is_accessibility_valid": accessibility_valid,
        "is_advanced_valid": not errors,
        "temporal_errors": temporal_errors,
        "accessibility_errors": accessibility_errors,
        "advanced_errors": errors,
        "next_agent": "plan_evaluation" if not errors else "advanced_repair",
    }