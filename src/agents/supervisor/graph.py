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

    def run_subgraph(name, app):
        def wrapper(state):
            result = app.invoke(dict(state))
            if not isinstance(result, dict):
                result = {}
            completed = list(state.get("completed_steps", []))
            if name not in completed:
                completed.append(name)
            return {
                **result,
                "completed_steps": completed,
                "handoff_count": int(state.get("handoff_count", 0)) + 1,
                "target_subgraph": name,
                "next_agent": name,
            }
        return wrapper

    def run_planner(state):
        from src.agents.planner.graph import planner_app
        return run_subgraph("planner", planner_app)(state)

    def run_tourist_help(state):
        from src.agents.tourist_help.graph import tourist_help_app
        return run_subgraph("tourist_help", tourist_help_app)(state)

    def run_execution(state):
        from src.agents.execution.graph import execution_app
        return run_subgraph("execution", execution_app)(state)

    def run_feedback(state):
        from src.agents.feedback.graph import feedback_app
        return run_subgraph("feedback", feedback_app)(state)

    # 1. Add nodes
    workflow.add_node("supervisor", supervisor_router_node)
    workflow.add_node("out_of_scope", out_of_scope_responder_node)
    workflow.add_node("profiling", profiling_agent_node)

    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("hydrate_context", hydrate_context_node)
    workflow.add_node("planner", run_planner)
    workflow.add_node("tourist_help", run_tourist_help)
    workflow.add_node("execution", run_execution)
    workflow.add_node("feedback", run_feedback)

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
            "planner": "planner",
            "tourist_help": "tourist_help",
            "execution": "execution",
            "feedback": "feedback",
            "FINISH": END,
        }
    )
    workflow.add_edge("planner", "supervisor")
    workflow.add_edge("tourist_help", "supervisor")
    workflow.add_edge("execution", "supervisor")
    workflow.add_edge("feedback", "supervisor")
    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("profiling", END)

    return workflow.compile()