"""
Planner Subgraph Implementation (Story 3.1 - 3.2 / Week 3).

Assembles draft generation with cyclic budget and logistics validation.
"""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.agents.planner.constraint_validator import constraint_validator_node
from src.agents.planner.draft_generator import draft_generator_node
from src.agents.planner.repairs import constraint_repair_node, repair_fallback_node


def route_validation(state: Dict[str, Any]) -> str:
    """Route constraint validation to completion, repair, or fallback."""
    if state.get("is_constraint_valid") is True:
        return "valid"
    if state.get("repair_attempts", 0) < 3:
        return "repair"
    return "fallback"


def create_planner_graph():
    """Build and compile the Week 3 Planner Part 1 workflow."""
    workflow = StateGraph(dict)

    workflow.add_node("draft_generator", draft_generator_node)
    workflow.add_node("constraint_validator", constraint_validator_node)
    workflow.add_node("constraint_repair", constraint_repair_node)
    workflow.add_node("repair_fallback", repair_fallback_node)

    workflow.add_edge(START, "draft_generator")
    workflow.add_edge("draft_generator", "constraint_validator")
    workflow.add_conditional_edges(
        "constraint_validator",
        route_validation,
        {
            "valid": END,
            "repair": "constraint_repair",
            "fallback": "repair_fallback",
        },
    )
    workflow.add_edge("constraint_repair", "constraint_validator")
    workflow.add_edge("repair_fallback", END)

    return workflow.compile()


planner_app = create_planner_graph()