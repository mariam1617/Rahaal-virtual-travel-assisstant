from typing import Annotated, Any, Dict, List, Literal, Optional, NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import  BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from src.state.schema import UserPreferences


class SupervisorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: NotRequired[str]
    user_query: NotRequired[str]
    user_preferences: NotRequired[UserPreferences]
    context_loaded: NotRequired[bool]
    context_warning: NotRequired[str]
    needs_profiling: NotRequired[bool]
    current_location: NotRequired[str]
    curr_location: NotRequired[str]
    location_details: NotRequired[Dict[str, Any]]
    next_agent: NotRequired[str]
    target_subgraph: NotRequired[str]
    intent: NotRequired[str]
    routing_reason: NotRequired[str] #UPDATED 21/11: Added routing_reason to SupervisorState for better traceability of routing decisions.
    confidence: NotRequired[float]
    is_egypt_scoped: NotRequired[bool]
    response: NotRequired[str]
    rejection_reason: NotRequired[str]
    completed_steps: NotRequired[List[str]]
    handoff_count: NotRequired[int]
    max_handoff_count: NotRequired[int]
    current_plan: NotRequired[Any]
    trip_requirements: NotRequired[Any]

# The target subgraphs and outcomes recognized by the Supervisor
TargetSubgraph = Literal[
    "planner",       # Multi-day itineraries, custom tour plans, activity scheduling
    "tourist_help",  # Visas, emergency numbers, local transport rules, nearby recommendations
    "execution",     # Booking flights/buses/trains, calendar syncing
    "feedback",      # Modifying existing confirmed plans, rating past activities
    "profiling",     # Onboarding, gathering preferences, handling vague/ambiguous intents
    "out_of_scope",  # Requests outside Egypt or non-travel queries
    "FINISH"         # Supervisor explicitly satisfied; stop the graph
]


class SupervisorDecision(BaseModel):
    """Structured output for the Supervisor Router Agent decision."""
    target: TargetSubgraph = Field(
        ...,
        description=(
            "The designated target subgraph for the user query: "
            "'planner' for planning/itineraries, "
            "'tourist_help' for immediate help/visas/emergency/local spots, "
            "'execution' for booking/calendar, "
            "'feedback' for itinerary adjustments/feedback, "
            "'profiling' for ambiguous/vague input needing interview, "
            "'out_of_scope' for non-Egypt travel or prohibited queries, "
            "'FINISH' when the request is satisfied or no further subgraph is needed."
        )
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the intent classification from 0.0 to 1.0."
    )
    is_egypt_scoped: bool = Field(
        ...,
        description=(
            "True if the query pertains to travel within or to Egypt (or is a general greeting). "
            "False if the user requests travel to other countries (e.g. Paris, Tokyo, Dubai)."
        )
    )
    reasoning: str = Field(
        ...,
        description="Concise rationale explaining the classification and scope verification."
    )
