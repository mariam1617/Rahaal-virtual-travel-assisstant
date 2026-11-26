"""Execution Subgraph for mock itinerary booking."""

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.execution.booking import (
    booking_fallback_node,
    route_booking,
    search_booking_node,
)
from src.agents.execution.calendar_sync import calendar_sync_node
from src.state.app_state import AppState


def create_execution_graph(checkpointer=None):
    workflow = StateGraph(AppState)
    workflow.add_node("SearchBooking", search_booking_node)
    workflow.add_node("booking_fallback", booking_fallback_node)
    workflow.add_node("CalendarSync", calendar_sync_node)
    workflow.add_edge(START, "SearchBooking")
    workflow.add_conditional_edges(
        "SearchBooking",
        route_booking,
        {
            "confirmed": "CalendarSync",
            "retry": "SearchBooking",
            "blocked": END,
            "fallback": "booking_fallback",
        },
    )
    workflow.add_edge("booking_fallback", END)
    workflow.add_edge("CalendarSync", END)
    return workflow.compile(checkpointer=checkpointer)


execution_app = create_execution_graph(checkpointer=checkpointer)