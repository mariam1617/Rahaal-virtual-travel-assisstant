"""
Tourist Help Subgraph Implementation (Story 2.1 - 2.4 / Week 2)
Assembles the complete Tourist Help workflow connecting intent resolution,
emergency RAG, advisory RAG, and the Help Verification Guardrail Node.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from src.agents.tourist_help.intent_resolver import intent_resolver_node
from src.agents.tourist_help.emergency import emergency_rag_node
from src.agents.tourist_help.advisories import advisories_rag_node
from src.agents.tourist_help.recommendations import recommendations_node
from src.agents.tourist_help.verification import help_verification_node


def route_help_subintent(state: Dict[str, Any]) -> str:
    """Conditional edge: routes to the specific Help RAG node based on resolved intent."""
    domain = state.get("help_domain", "general")
    if domain == "emergency":
        return "emergency_rag"
    elif domain == "advisories":
        return "advisories_rag"
    elif domain == "recommendations":
        return "recommendations"
    return "advisories_rag"


def create_tourist_help_graph():
    """
    Constructs and compiles the Tourist Help Subgraph with verification guardrail.
    Workflow:
        START -> intent_resolver -> [emergency_rag | advisories_rag | recommendations]
                                 -> verification -> END
    """
    workflow = StateGraph(dict)

    # 1. Add functional nodes
    workflow.add_node("intent_resolver", intent_resolver_node)
    workflow.add_node("emergency_rag", emergency_rag_node)
    workflow.add_node("advisories_rag", advisories_rag_node)
    workflow.add_node("recommendations", recommendations_node)
    workflow.add_node("verification", help_verification_node)

    # 2. Add entry point & routing edges
    workflow.add_edge(START, "intent_resolver")
    workflow.add_conditional_edges(
        "intent_resolver",
        route_help_subintent,
        {
            "emergency_rag": "emergency_rag",
            "advisories_rag": "advisories_rag",
            "recommendations": "recommendations"
        }
    )

    # 3. All paths must pass through the verification guardrail before finishing
    workflow.add_edge("emergency_rag", "verification")
    workflow.add_edge("advisories_rag", "verification")
    workflow.add_edge("recommendations", "verification")
    workflow.add_edge("verification", END)

    return workflow.compile()


# Pre-compiled application instance for direct invocation
tourist_help_app = create_tourist_help_graph()
