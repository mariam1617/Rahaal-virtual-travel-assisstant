from langgraph.graph import END, START, StateGraph

from src.agents.feedback.profiling import profiling_agent_node
from src.agents.supervisor.context import hydrate_context_node, route_context
from src.agents.supervisor.router import (
    out_of_scope_responder_node,
    supervisor_conditional_edge,
    supervisor_router_node,
)
from src.agents.supervisor.state import SupervisorState
from src.agents.tourist_help.intent_resolver import resolve_location_node
def create_supervisor_graph():
    """
    Build and compile the Supervisor LangGraph workflow with conditional routing
    and fallback edges.
    """
    # from langgraph.graph import StateGraph, START, END
    # from src.agents.feedback.profiling import profiling_agent_node
    # from src.agents.supervisor.context import hydrate_context_node, route_context
    # from src.agents.tourist_help.intent_resolver import resolve_location_node

    workflow = StateGraph(SupervisorState)

    # 1. Add nodes
    workflow.add_node("supervisor", supervisor_router_node)
    workflow.add_node("out_of_scope", out_of_scope_responder_node)
    workflow.add_node("profiling", profiling_agent_node)

    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("hydrate_context", hydrate_context_node)

    # 2. Add edges & conditional routing
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
            "planner": END,
            "tourist_help": END,
            "execution": END,
            "feedback": END,
        }
    )
    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("profiling", END)

    return workflow.compile()