"""
Planner Subgraph Implementation (Story 3.1 - 3.2 / Week 3).

Assembles draft generation with cyclic budget and logistics validation.
"""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.agents.planner.constraint_validator import constraint_validator_node
from src.agents.planner.draft_generator import draft_generator_node
from src.agents.planner.repairs import (
    advanced_repair_fallback_node,
    advanced_repair_node,
    constraint_repair_node,
    repair_fallback_node,
)
from src.agents.planner.evaluator import plan_evaluation_fallback_node, plan_evaluation_node
from src.agents.planner.temporal_validator import temporal_accessibility_validator_node


def route_validation(state: Dict[str, Any]) -> str:
    """Route constraint validation to completion, repair, or fallback."""
    if state.get("is_constraint_valid") is True:
        return "valid"
    if state.get("repair_attempts", 0) < 3:
        return "repair"
    return "fallback"


def route_advanced_validation(state: Dict[str, Any]) -> str:
    """Route advanced validation to evaluation, repair, or fallback."""
    if state.get("is_advanced_valid") is True:
        return "valid"
    if state.get("advanced_repair_attempts", 0) < 3:
        return "repair"
    return "fallback"


def create_planner_graph():
    """Build and compile the Week 3 Planner Part 1 workflow."""
    workflow = StateGraph(dict)

    workflow.add_node("draft_generator", draft_generator_node)
    workflow.add_node("constraint_validator", constraint_validator_node)
    workflow.add_node("constraint_repair", constraint_repair_node)
    workflow.add_node("repair_fallback", repair_fallback_node)
    workflow.add_node("temporal_accessibility_validator", temporal_accessibility_validator_node)
    workflow.add_node("advanced_repair", advanced_repair_node)
    workflow.add_node("advanced_repair_fallback", advanced_repair_fallback_node)
    workflow.add_node("plan_evaluation", plan_evaluation_node)
    workflow.add_node("evaluation_fallback", plan_evaluation_fallback_node)

    workflow.add_edge(START, "draft_generator")
    workflow.add_edge("draft_generator", "constraint_validator")
    workflow.add_conditional_edges(
        "constraint_validator",
        route_validation,
        {
            "valid": "temporal_accessibility_validator",
            "repair": "constraint_repair",
            "fallback": "repair_fallback",
        },
    )
    workflow.add_edge("constraint_repair", "constraint_validator")
    workflow.add_edge("repair_fallback", END)
    workflow.add_conditional_edges(
        "temporal_accessibility_validator",
        route_advanced_validation,
        {
            "valid": "plan_evaluation",
            "repair": "advanced_repair",
            "fallback": "advanced_repair_fallback",
        },
    )
    workflow.add_edge("advanced_repair", "temporal_accessibility_validator")
    workflow.add_edge("advanced_repair_fallback", END)
    workflow.add_conditional_edges(
        "plan_evaluation",
        lambda state: "good" if state.get("plan_quality") == "Good" else "poor",
        {"good": END, "poor": "evaluation_fallback"},
    )
    workflow.add_edge("evaluation_fallback", END)

    return workflow.compile()


planner_app = create_planner_graph()