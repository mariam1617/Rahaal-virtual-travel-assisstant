"""
FastAPI Router for Feedback & Memory Subgraph (Epic 5)
Provides chat entry points for active itinerary modifications and long-term preference learning.
Module: src/api/feedback.py
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.agents.feedback.graph import feedback_app
from src.state.schema import GeneratedPlan, TripRequirements, UserPreferences
from src.db.profile_store import ProfileStore


router = APIRouter(prefix="/feedback", tags=["Feedback & Memory Agent"])


class FeedbackAPIRequest(BaseModel):
    message: str = Field(..., description="User feedback or trip adjustment request.")
    user_id: str = Field(default="test_user", description="Identifier of the active user.")
    current_plan: Optional[Dict[str, Any]] = Field(default=None, description="Active generated itinerary, if available.")
    trip_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Current trip constraints.")


class FeedbackAPIResponse(BaseModel):
    feedback_type: str
    feedback_confidence: float
    response: str
    current_plan: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    is_executable: bool = True
    next_agent: str = "feedback_completed"
    plan_modification: Optional[Dict[str, Any]] = None
    extracted_insights: List[Dict[str, Any]] = Field(default_factory=list)


@router.post("/process", response_model=FeedbackAPIResponse)
def process_user_feedback(request: FeedbackAPIRequest) -> FeedbackAPIResponse:
    """
    Classifies feedback, modifies active itinerary (with Planner validation loops),
    and updates long-term memory (LTM) profile store.
    """
    plan_obj = None
    if request.current_plan:
        plan_obj = GeneratedPlan(**request.current_plan)

    req_obj = None
    if request.trip_requirements:
        req_obj = TripRequirements(**request.trip_requirements)

    profile_obj = ProfileStore.get_user_profile(request.user_id) or ProfileStore.ensure_user_profile(request.user_id)

    input_state = {
        "user_id": request.user_id,
        "messages": [("user", request.message)],
        "current_plan": plan_obj,
        "trip_requirements": req_obj,
        "user_preferences": profile_obj,
    }

    result = feedback_app.invoke(
        input_state,
        config={"configurable": {"thread_id": f"feedback-{request.user_id}"}}
    )

    plan_res = result.get("current_plan")
    pref_res = result.get("user_preferences")

    return FeedbackAPIResponse(
        feedback_type=result.get("feedback_type", "unknown"),
        feedback_confidence=result.get("feedback_confidence", 0.0),
        response=result.get("response", ""),
        current_plan=plan_res.model_dump() if isinstance(plan_res, GeneratedPlan) else (plan_res or None),
        user_preferences=pref_res.model_dump() if isinstance(pref_res, UserPreferences) else (pref_res or None),
        is_executable=result.get("is_executable", True),
        next_agent=result.get("next_agent", "feedback_completed"),
        plan_modification=result.get("plan_modification"),
        extracted_insights=result.get("extracted_insights", []),
    )
