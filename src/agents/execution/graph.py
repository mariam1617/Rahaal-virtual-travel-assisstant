"""Execution Subgraph for mock itinerary booking."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.execution.booking import (
    booking_fallback_node,
    route_booking,
    search_booking_node,
)


def create_execution_graph(checkpointer=None):
    workflow = StateGraph(dict)
    workflow.add_node("SearchBooking", search_booking_node)
    workflow.add_node("booking_fallback", booking_fallback_node)
    workflow.add_edge(START, "SearchBooking")
    workflow.add_conditional_edges(
        "SearchBooking",
        route_booking,
        {
            "confirmed": END,
            "retry": "SearchBooking",
            "blocked": END,
            "fallback": "booking_fallback",
        },
    )
    workflow.add_edge("booking_fallback", END)
    return workflow.compile(checkpointer=checkpointer)


execution_app = create_execution_graph(checkpointer=MemorySaver())