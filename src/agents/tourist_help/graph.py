"""
Tourist Help Subgraph Definition & Workflow Compilation (Story 2.1)
Module: src/agents/tourist_help/graph.py
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from src.agents.tourist_help.intent_resolver import (
    resolve_location_node,
    intent_parser_node,
    route_request,
    fallback_agent_node,
)
from src.agents.tourist_help.emergency import emergency_agent_node
from src.agents.tourist_help.advisories import advisories_agent_node
from src.agents.tourist_help.recommendations import recommendations_agent_node


def planner_delegate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delegate node routing to the Planner Agent when a trip-planning intent is detected.
    """
    return {
        **state,
        "next_agent": "planner",
        "response": "Forwarding your itinerary planning request to the Planner Agent."
    }


def create_tourist_help_graph():
    """
    Builds and compiles the synchronous Tourist Help LangGraph workflow.
    
    Execution flow:
      START
        │
        ▼
      resolve_location_node (Reverse geocodes GPS coordinates into current_location)
        │
        ▼
      intent_parser_node (Parses inquiry and updates state["current_intent"])
        │
        ▼ [conditional edge: route_request]
      ├── "emergency"       -> emergency_agent_node
      ├── "advisories"      -> advisories_agent_node
      ├── "recommendations" -> recommendations_agent_node
      ├── "planner_agent"   -> planner_delegate_node
      └── "fallback"        -> fallback_agent_node
        │
        ▼
       END
    """
    workflow = StateGraph(dict)

    # 1. Register Nodes
    workflow.add_node("resolve_location_node", resolve_location_node)
    workflow.add_node("intent_parser_node", intent_parser_node)
    workflow.add_node("emergency", emergency_agent_node)
    workflow.add_node("advisories", advisories_agent_node)
    workflow.add_node("recommendations", recommendations_agent_node)
    workflow.add_node("planner_agent", planner_delegate_node)
    workflow.add_node("fallback", fallback_agent_node)

    # 2. Add Linear Entry Edges
    workflow.add_edge(START, "resolve_location_node")
    workflow.add_edge("resolve_location_node", "intent_parser_node")

    # 3. Add Conditional Routing Edge
    workflow.add_conditional_edges(
        "intent_parser_node",
        route_request,
        {
            "emergency": "emergency",
            "advisories": "advisories",
            "recommendations": "recommendations",
            "planner_agent": "planner_agent",
            "fallback": "fallback",
        }
    )

    # 4. Terminal Edges
    workflow.add_edge("emergency", END)
    workflow.add_edge("advisories", END)
    workflow.add_edge("recommendations", END)
    workflow.add_edge("planner_agent", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()

