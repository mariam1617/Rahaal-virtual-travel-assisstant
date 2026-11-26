"""Final quality evaluation for the Planner Subgraph."""

from typing import Any, Dict

from langchain_core.messages import AIMessage

from src.state.schema import GeneratedPlan


def evaluate_plan(plan: GeneratedPlan, errors: list[str] | None = None) -> Dict[str, Any]:
    """Return a deterministic quality grade and rationale."""
    remaining_errors = errors or []
    score = 100 if not remaining_errors else max(0, 100 - len(remaining_errors) * 20)
    grade = "Good" if score >= 70 else "Poor"
    return {
        "score": score,
        "grade": grade,
        "rationale": "All planner validations passed." if grade == "Good" else "Validation issues remain.",
    }


def plan_evaluation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the validated itinerary and expose the result in graph state."""
    plan = state.get("current_plan")
    evaluation = evaluate_plan(plan, state.get("advanced_errors", [])) if isinstance(plan, GeneratedPlan) else {
        "score": 0,
        "grade": "Poor",
        "rationale": "No active plan available for evaluation.",
    }
    is_executable = evaluation["grade"] == "Good"
    response = (
        "Your itinerary passed the final quality evaluation and is ready for execution."
        if is_executable
        else "We generated a best-effort itinerary, but it needs review before execution."
    )
    updated_plan = (
        plan.model_copy(update={"status": "approved" if is_executable else "review"})
        if isinstance(plan, GeneratedPlan)
        else plan
    )
    return {
        "evaluation": evaluation,
        "plan_quality": evaluation["grade"],
        "is_executable": is_executable,
        "current_plan": updated_plan,
        "response": response,
        "messages": [AIMessage(content=response)],
        "next_agent": "planner_completed" if is_executable else "evaluation_fallback",
    }


def plan_evaluation_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Expose a best-effort plan with an explicit non-executable warning."""
    plan = state.get("current_plan")
    updated_plan = (
        plan.model_copy(update={"status": "review"})
        if isinstance(plan, GeneratedPlan)
        else plan
    )
    response = (
        "⚠️ **Plan Review Required**\n\n"
        "This is a best-effort itinerary. It is not approved for execution "
        "because the final quality evaluation was Poor."
    )
    return {
        "current_plan": updated_plan,
        "is_executable": False,
        "response": response,
        "messages": [AIMessage(content=response)],
        "next_agent": "evaluation_fallback_completed",
    }