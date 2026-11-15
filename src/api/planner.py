"""Planner API endpoint for synchronous itinerary generation and validation."""

from fastapi import APIRouter

from src.agents.planner.graph import planner_app
from src.api.models import PlannerRequest, PlannerResponse
from src.state.schema import TripRequirements, UserPreferences


router = APIRouter(prefix="/planner", tags=["Planner Agent"])


@router.post("/plan", response_model=PlannerResponse)
def create_plan(request: PlannerRequest) -> PlannerResponse:
    """Generate, validate, repair, and evaluate a planner itinerary."""
    preferences = UserPreferences(
        user_id="api_user",
        name="API User",
        nationality="unknown",
        accessibility_needs=request.accessibility_needs,
    )
    result = planner_app.invoke(
        {
            "messages": [("user", f"Plan {request.duration_days} days in {request.destination}")],
            "trip_requirements": TripRequirements(
                destination=request.destination,
                duration_days=request.duration_days,
                total_budget=request.total_budget,
            ),
            "user_preferences": preferences,
            "repair_attempts": 0,
            "advanced_repair_attempts": 0,
            "errors": [],
            "constraint_errors": [],
        }
    )
    plan = result.get("current_plan")
    return PlannerResponse(
        plan=plan.model_dump() if plan else {},
        is_constraint_valid=result.get("is_constraint_valid", False),
        is_advanced_valid=result.get("is_advanced_valid", False),
        is_executable=result.get("is_executable", False),
        evaluation=result.get("evaluation", {}),
        response=result.get("response", ""),
        temporal_errors=result.get("temporal_errors", []),
        accessibility_errors=result.get("accessibility_errors", []),
        errors=result.get("advanced_errors", []) + result.get("constraint_errors", []),
    )
