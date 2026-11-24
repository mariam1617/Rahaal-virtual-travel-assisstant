"""Unified graph state for Ra7al: the single source of truth.

Rules:
- Only `messages` has a reducer. Every other key is plain overwrite.
- Nodes return only the keys they change (never `{**state, ...}` in new code).
- Nodes return only NEW messages: {"messages": [ai_msg]}.
"""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.state.schema import GeneratedPlan, TripRequirements, UserPreferences


class AppState(TypedDict, total=False):
    # ---- Conversation & identity ----
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    session_id: str
    is_guest: bool                      # reserved (Dev 1: anonymous user isolation)
    user_query: str                     # DEPRECATED: read the last human message instead
    response: str                       # final user-facing text for this turn

    # ---- Long-term memory / context hydration ----
    user_preferences: Optional[UserPreferences]
    context_loaded: bool
    context_warning: str
    needs_profiling: bool
    missing_constraints: List[str]

    # ---- Current trip (session) ----
    trip_requirements: Optional[TripRequirements]
    current_plan: Optional[GeneratedPlan]

    # ---- Supervisor routing ----
    next_agent: str
    target_subgraph: str
    handoff_to: str                     # reserved (Dev 2/3/4): explicit hand-off between subgraphs
    intent: str
    routing_reason: str
    confidence: float
    is_egypt_scoped: bool
    rejection_reason: str
    completed_steps: List[str]
    handoff_count: int
    max_handoff_count: int

    # ---- Location (Story 2.1) ----
    raw_coordinates: Optional[Dict[str, float]]
    coordinates: Any
    current_location: str               # canonical formatted address string
    curr_location: str                  # DEPRECATED alias of current_location
    location_details: Dict[str, Any]    # structured: city, governorate, neighborhood, country_code, lat, lon, error

    # ---- Tourist Help / RAG / verification (Stories 2.x) ----
    current_intent: str
    help_domain: str
    next_help_node: str
    candidate_response: str
    retrieved_context: str
    is_verified: bool
    verified_response: str
    verification_result: Dict[str, Any]
    pending_question: str               # reserved (Dev 3): e.g. ask nationality
    retrieval_sources: List[Dict[str, Any]]  # reserved (Dev 3)
    context: str                        # fallback for retrieved_context in verification
    draft_response: str                 # fallback for candidate_response in verification
    intent_category: str                # fallback for help_domain in verification
    rag_docs: Any                       # fallback for retrieved_context in verification

    # ---- Planner (Stories 3.x) ----
    is_constraint_valid: bool
    constraint_errors: List[str]
    repair_attempts: int
    budget_overage: float
    total_cost: float
    is_temporal_valid: bool
    is_accessibility_valid: bool
    is_advanced_valid: bool
    temporal_errors: List[str]
    accessibility_errors: List[str]
    advanced_errors: List[str]
    advanced_repair_attempts: int
    evaluation: Dict[str, Any]
    plan_quality: str
    is_executable: bool

    # ---- Approval / execution (Stories 4.x) ----
    plan_version: str                   # reserved (Dev 2)
    approved_plan_version: str          # reserved (Dev 2)
    approval_status: str                # "approved" | "rejected" | "pending"
    approval_required: bool
    booking_status: str
    booking_attempts: int
    bookings: List[Dict[str, Any]]
    booking_errors: List[str]
    calendar_status: str
    calendar_events: List[Dict[str, Any]]
    calendar_errors: List[str]
    errors: List[str]
    mock_booking_failure: bool          # test mock flag
    mock_booking_failures_remaining: Optional[int]  # test mock flag
    mock_calendar_failure: bool         # test mock flag

    # ---- Feedback & memory (Stories 5.x) ----
    feedback_type: str
    feedback_confidence: float
    feedback_reasoning: str
    feedback_classification: Dict[str, Any]
    extracted_insights: List[Dict[str, Any]]
    pending_preferences: Dict[str, Any]
    plan_modification: Dict[str, Any]
    greeted: bool                       # reserved (Dev 4): greet only once
