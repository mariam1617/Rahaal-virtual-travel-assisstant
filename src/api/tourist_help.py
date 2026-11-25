"""
Tourist Help API Router (Story 2.1 - 2.4 / Week 2)
FastAPI endpoint executing the compiled Tourist Help Subgraph synchronously.
Module: src/api/tourist_help.py
"""

from typing import Optional, Dict, Any, Union, List
from fastapi import APIRouter
from pydantic import BaseModel

from src.runner import execute_graph_sync
from src.agents.tourist_help.graph import tourist_help_app

router = APIRouter(prefix="/tourist-help", tags=["Tourist Help Agent"])


class TouristHelpRequest(BaseModel):
    user_id: str = "traveler_101"
    message: str
    location: Optional[Union[Dict[str, float], str]] = None


class TouristHelpResponse(BaseModel):
    response: str
    help_domain: str
    is_verified: bool
    verification_details: Dict[str, Any]
    current_location: Optional[str] = None
    sources: List[Dict[str, Any]] = []          # chunks the answer was built from (title / section / url)
    pending_question: Optional[str] = None      # e.g. "Which passport do you hold?" (answer in the next turn)


@router.post("/query", response_model=TouristHelpResponse)
def query_tourist_help(request: TouristHelpRequest):
    """
    Direct synchronous entry point for Tourist Help queries (emergency, advisories, recommendations).
    Runs through location resolution, intent parsing, RAG pipelines, and the Help Verification Guardrail.
    """
    raw_coords = request.location if isinstance(request.location, dict) else None
    current_loc = request.location if isinstance(request.location, str) else None

    initial_state = {
        "user_id": request.user_id,
        "messages": [("user", request.message)],
        "user_query": request.message,
        "raw_coordinates": raw_coords,
        "current_location": current_loc,
        "current_intent": "unclassified"
    }

    final_state = execute_graph_sync(tourist_help_app, initial_state)

    return TouristHelpResponse(
        response=final_state.get("verified_response", final_state.get("response", "")),
        help_domain=final_state.get("help_domain", "general"),
        is_verified=final_state.get("is_verified", False),
        verification_details=final_state.get("verification_result", {}),
        current_location=final_state.get("current_location"),
        sources=final_state.get("retrieval_sources") or [],
        pending_question=final_state.get("pending_question") or None,
    )
