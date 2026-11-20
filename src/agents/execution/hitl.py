"""Human approval gate for executable itineraries."""

from typing import Any, Dict

from langgraph.types import interrupt

from src.state.schema import GeneratedPlan


def human_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pause before execution until the user explicitly approves the plan."""
    plan = state.get("current_plan")
    if not isinstance(plan, GeneratedPlan):
        return {
            **state,
            "is_executable": False,
            "approval_status": "rejected",
            "response": "No executable itinerary is available for approval.",
            "next_agent": "human_approval_completed",
        }

    decision = interrupt(
        {
            "type": "plan_approval",
            "message": "Review this itinerary and explicitly approve it before execution.",
            "plan": plan.model_dump(),
        }
    )
    approved = decision is True or (
        isinstance(decision, dict) and decision.get("approved") is True
    )

    if approved:
        return {
            **state,
            "approval_status": "approved",
            "approval_required": False,
            "response": "Your itinerary was approved and is ready for execution.",
            "next_agent": "execution_pending",
        }

    plan.status = "review"
    return {
        **state,
        "current_plan": plan,
        "is_executable": False,
        "approval_status": "rejected",
        "approval_required": False,
        "response": "The itinerary was not approved and no execution actions were taken.",
        "next_agent": "human_approval_completed",
    }