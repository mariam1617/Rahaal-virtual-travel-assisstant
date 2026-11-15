"""
Tourist Help API Router (Story 2.1 - 2.4 / Week 2)
FastAPI endpoint executing the compiled Tourist Help Subgraph synchronously.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

from src.runner import execute_graph_sync
from src.agents.tourist_help.graph import tourist_help_app

router = APIRouter(prefix="/tourist-help", tags=["Tourist Help Agent"])


class TouristHelpRequest(BaseModel):
    user_id: str = "traveler_101"
    message: str
    location: Optional[Dict[str, float]] = None


class TouristHelpResponse(BaseModel):
    response: str
    help_domain: str
    is_verified: bool
    verification_details: Dict[str, Any]


@router.post("/query", response_model=TouristHelpResponse)
def query_tourist_help(request: TouristHelpRequest):
    """
    Direct synchronous entry point for Tourist Help queries (emergency, advisories, recommendations).
    Runs through intent resolution, RAG pipelines, and the Help Verification Guardrail.
    """
    initial_state = {
        "user_id": request.user_id,
        "messages": [("user", request.message)],
        "user_query": request.message,
        "current_location": request.location
    }

    final_state = execute_graph_sync(tourist_help_app, initial_state)

    return TouristHelpResponse(
        response=final_state.get("verified_response", final_state.get("response", "")),
        help_domain=final_state.get("help_domain", "general"),
        is_verified=final_state.get("is_verified", False),
        verification_details=final_state.get("verification_result", {})
    )
