"""
Feedback & Memory Subgraph Assembly (Epic 5)
Story 5.2: Active Itinerary Modification
Story 5.3: Long-Term Preference Learning
Module: src/agents/feedback/graph.py
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from src.agents.feedback.router import feedback_router_node, route_feedback
from src.agents.feedback.itinerary_update import itinerary_modifier_node
from src.agents.feedback.preference_learning import (
    insight_extractor_node,
    ltm_preference_update_node,
)
from src.state.app_state import AppState


def feedback_fallback_node(state: AppState) -> Dict[str, Any]:
    msg = (
        "I couldn't quite understand whether you'd like to adjust your current itinerary "
        "or set a personal travel preference. Could you please specify?\n\n"
        "• For trip adjustments: e.g., 'Change Day 1 dinner to an Italian restaurant'\n"
        "• For personal preferences: e.g., 'I prefer morning flights' or 'I am vegan'"
    )
    ai_msg = AIMessage(content=msg)
    return {
        "response": msg,
        "messages": [ai_msg],
        "next_agent": "feedback_completed",
    }
     

def route_after_ltm(state: AppState) -> str:
    """
    Routes from LTM Updater: if compound ('both'), proceeds to Active Itinerary Modifier,
    otherwise ends execution.
    """
    fb_type = state.get("feedback_type")
    if fb_type == "both":
        return "itinerary_modifier"
    return "end"


def create_feedback_graph(checkpointer=None):
    """
    Builds and compiles the Feedback & Memory LangGraph workflow.
    """
    workflow = StateGraph(AppState)
    # 1. Add Subgraph Nodes
    workflow.add_node("feedback_router", feedback_router_node)
    workflow.add_node("itinerary_modifier", itinerary_modifier_node)
    workflow.add_node("insight_extractor", insight_extractor_node)
    workflow.add_node("ltm_updater", ltm_preference_update_node)
    workflow.add_node("feedback_fallback", feedback_fallback_node)

    # 2. Add Workflow Edges
    workflow.add_edge(START, "feedback_router")

    # Conditional edge from Router
    workflow.add_conditional_edges(
        "feedback_router",
        route_feedback,
        {
            "itinerary_modifier": "itinerary_modifier",
            "insight_extractor": "insight_extractor",
            "feedback_fallback": "feedback_fallback",
        }
    )

    # Edge from Insight Extractor to LTM Updater
    workflow.add_edge("insight_extractor", "ltm_updater")

    # Conditional edge from LTM Updater (handles compound intent)
    workflow.add_conditional_edges(
        "ltm_updater",
        route_after_ltm,
        {
            "itinerary_modifier": "itinerary_modifier",
            "end": END,
        }
    )

    # Termination edges
    workflow.add_edge("itinerary_modifier", END)
    workflow.add_edge("feedback_fallback", END)

    return workflow.compile(checkpointer=checkpointer)


feedback_app = create_feedback_graph()
