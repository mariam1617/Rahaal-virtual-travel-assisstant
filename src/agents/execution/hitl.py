"""Human approval gate for executable itineraries."""

from typing import Any, Dict

from langgraph.types import interrupt

from src.state.schema import GeneratedPlan


def human_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pause before execution until the user explicitly approves the plan."""
    # TODO (Problem 7): Approval is not connected to anything:
    # 1. interrupt() requires a checkpointer on the graph that actually runs. The supervisor graph has none,
    #    and the planner's MemorySaver is separate from the parent's thread.
    # 2. On resume, a node re-runs from its start. Because the planner is called through app.invoke()
    #    inside a wrapper node, resuming likely re-runs the whole planner or fails to find paused state.
    # 3. After approval it returns next_agent: "execution_pending", which no graph reads. Nothing triggers Execution!
    # 4. A rejection ends the flow, although the user may just want a change (e.g. route to feedback/modifier).
    # 5. Approving is not tied to a plan version. If the user edits the plan after the approval card was shown,
    #    the old approval could apply to the new plan. Store plan_version / plan_id with the approval token.
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