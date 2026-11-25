"""
Tourist Help Subgraph Implementation (Story 2.1 - 2.4 / Week 2)
Assembles the complete Tourist Help workflow connecting intent resolution,
location geocoding, emergency RAG, advisory RAG, recommendations,
and the Help Verification Guardrail Node.
Module: src/agents/tourist_help/graph.py
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from src.state import AppState  

from src.agents.tourist_help.intent_resolver import (
    resolve_location_node,
    intent_parser_node,
    route_request,
    fallback_agent_node,
)
from src.agents.tourist_help.emergency import emergency_rag_node
from src.agents.tourist_help.advisories import advisories_rag_node
from src.agents.tourist_help.recommendations import recommendations_node
from src.agents.tourist_help.verification import help_verification_node


def planner_delegate_node(state: AppState) -> Dict[str, Any]:
    return {
        "next_agent": "planner",
        "response": "Forwarding your itinerary planning request to the Planner Agent.",
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
      intent_parser_node (Parses inquiry, sets state["current_intent"] & state["help_domain"])
        │
        ▼ [conditional edge: route_request]
      ├── "emergency"       -> emergency_rag_node    ──────┐
      ├── "advisories"      -> advisories_rag_node   ──────┼──► verification (Guardrail Node) -> END
      ├── "recommendations" -> recommendations_node  ──────┘
      ├── "planner_agent"   -> planner_delegate_node ─────────► END
      └── "fallback"        -> fallback_agent_node   ─────────► END
    """
    workflow = StateGraph(AppState)

    # 1. Register functional nodes
    workflow.add_node("resolve_location_node", resolve_location_node)
    workflow.add_node("intent_parser_node", intent_parser_node)
    workflow.add_node("emergency", emergency_rag_node)
    workflow.add_node("advisories", advisories_rag_node)
    workflow.add_node("recommendations", recommendations_node)
    workflow.add_node("verification", help_verification_node)
    workflow.add_node("planner_agent", planner_delegate_node)
    workflow.add_node("fallback", fallback_agent_node)

    # 2. Add linear entry edges
    workflow.add_edge(START, "resolve_location_node")
    workflow.add_edge("resolve_location_node", "intent_parser_node")

    # 3. Add conditional edge from intent_parser_node
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

    # 4. RAG and recommendation outputs pass through verification guardrail
    workflow.add_edge("emergency", "verification")
    workflow.add_edge("advisories", "verification")
    workflow.add_edge("recommendations", "verification")
    workflow.add_edge("verification", END)

    # 5. Direct terminal edges for delegation and fallback
    workflow.add_edge("planner_agent", END)
    workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=checkpointer)

# Pre-compiled application instance for direct invocation
tourist_help_app = create_tourist_help_graph()
