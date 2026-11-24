from langgraph.graph import END, START, StateGraph

from src.agents.feedback.profiling import profiling_agent_node
from src.agents.supervisor.context import hydrate_context_node, route_context
from src.agents.supervisor.router import (
    out_of_scope_responder_node,
    supervisor_conditional_edge,
    supervisor_router_node,
)
from src.agents.tourist_help.intent_resolver import resolve_location_node
from src.state.app_state import AppState


def create_supervisor_graph(checkpointer=None):
    """
    Build and compile the Supervisor LangGraph workflow with conditional routing
    and fallback edges.
    """
    from src.agents.execution.graph import execution_app
    from src.agents.feedback.graph import feedback_app
    from src.agents.planner.graph import planner_app
    from src.agents.tourist_help.graph import tourist_help_app

    workflow = StateGraph(AppState)

    # 1. Add nodes (subgraphs mounted directly)
    workflow.add_node("supervisor", supervisor_router_node)
    workflow.add_node("out_of_scope", out_of_scope_responder_node)
    workflow.add_node("profiling", profiling_agent_node)
    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("hydrate_context", hydrate_context_node)

    workflow.add_node("planner", planner_app)
    workflow.add_node("tourist_help", tourist_help_app)
    workflow.add_node("execution", execution_app)
    workflow.add_node("feedback", feedback_app)

    # 2. Edges & conditional routing
    workflow.add_edge(START, "resolve_location")
    workflow.add_edge("resolve_location", "hydrate_context")
    workflow.add_conditional_edges(
        "hydrate_context",
        route_context,
        {"supervisor": "supervisor", "profiling": "profiling"},
    )
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_conditional_edge,
        {
            "out_of_scope": "out_of_scope",
            "profiling": "profiling",
            "planner": "planner",
            "tourist_help": "tourist_help",
            "execution": "execution",
            "feedback": "feedback",
            "FINISH": END,
        },
    )
    workflow.add_edge("planner", "supervisor")
    workflow.add_edge("tourist_help", "supervisor")
    workflow.add_edge("execution", "supervisor")
    workflow.add_edge("feedback", "supervisor")
    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("profiling", END)

    return workflow.compile(checkpointer=checkpointer)