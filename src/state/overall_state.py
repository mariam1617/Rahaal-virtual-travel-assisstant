from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator
from src.state.schema import UserPreferences, TripRequirements, GeneratedPlan
#UPDATE 21/11: 
# Added add_messages for LangGraph-managed message history.
# Updated overall_state.py to use add_messages
class OverallState(TypedDict):
    # مدخلات ومحادثة
    # messages: Annotated[List[Dict[str, Any]], operator.add]

    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    user_query: Optional[str]
    
    # الذاكرة الدائمة (Long-Term Memory)
    user_preferences: Optional[UserPreferences]
    context_loaded: Optional[bool]
    context_warning: Optional[str]
    needs_profiling: Optional[bool]
    
    # بيانات الرحلة الحالية (Short-Term Session)
    trip_requirements: Optional[TripRequirements]
    
    # مخرجات ومسودة الخطط التي تتنقل بين Planner / Validator / Supervisor
    current_plan: Optional[GeneratedPlan]
    
    # متغيرات التوجيه والتنفيذ
    next_agent: Optional[str]
    current_intent: Optional[str]
    help_domain: Optional[str]
    errors: Annotated[List[str], operator.add]
    booking_status: Optional[str]
    booking_attempts: int
    bookings: List[Dict[str, Any]]
    booking_errors: Annotated[List[str], operator.add]
    calendar_status: Optional[str]
    calendar_events: List[Dict[str, Any]]
    calendar_errors: Annotated[List[str], operator.add]

    # Geo location (Story 2.1)
    raw_coordinates: Optional[Dict[str, float]]
    current_location: Optional[str]
    location_details: Optional[Dict[str, Any]]

    # RAG & Verification (Story 2.2, Story 2.3, Story 2.4)
    candidate_response: Optional[str]
    retrieved_context: Optional[str]
    is_verified: Optional[bool]
    verified_response: Optional[str]
    verification_result: Optional[Dict[str, Any]]

    # Planner & Constraint Loop (Story 3.1, 3.2)
    is_constraint_valid: Optional[bool]
    constraint_errors: Annotated[List[str], operator.add]
    repair_attempts: int
    total_cost: Optional[float]

    # Feedback & Memory Subgraph (Story 5.2, 5.3)
    feedback_type: Optional[str]
    feedback_confidence: Optional[float]
    feedback_reasoning: Optional[str]
    extracted_insights: Optional[List[Dict[str, Any]]]
    pending_preferences: Optional[Dict[str, Any]]
    plan_modification: Optional[Dict[str, Any]]
