from typing import Literal
from pydantic import BaseModel, Field

from src.state.app_state import AppState

SupervisorState = AppState  # DEPRECATED alias


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
