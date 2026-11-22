"""Context hydration for Supervisor requests (Story 1.1)."""
#UPDATE 21/11: 
# Added SupervisorState as a TypedDict .

from typing import Any, Dict
from src.state.memory import MemoryLoader
from src.agents.supervisor.router import SupervisorState

# def hydrate_context_node(state: Dict[str, Any]) -> Dict[str, Any]:

def hydrate_context_node(state: SupervisorState) -> Dict[str, Any]:
    """Load known user preferences and preserve request/session context."""
    # TODO (Problem 13): All users share one profile if user_id falls back to a common test_user.
    # Ensure guest/anonymous sessions do not overwrite each other or share persistent profiles.
    user_id = state.get("user_id")
    if not user_id:
        return {**state, "context_loaded": False}

    try:
        # TODO (Problem 2): Profile logic exists three times (ProfileStore, MemoryLoader, SQLiteUserRepository).
        # Collapse into a unified ProfileService on top of SQLiteUserRepository with separate get and ensure methods.
        profile = MemoryLoader.get_user_profile(user_id)
    except Exception:
        return {
            **state,
            "context_loaded": False,
            "context_warning": (
                f"Unable to load profile for user '{user_id}' due to a "
                "database or connection error."
            ),
            "needs_profiling": True,
            "next_agent": "profiling",
        }

    if profile is None:
        return {
            **state,
            "context_loaded": False,
            "context_warning": f"No profile found for user '{user_id}'.",
            "needs_profiling": True,
            "next_agent": "profiling",
        }

    return {
        **state,
        "user_preferences": profile,
        "context_loaded": True,
        "needs_profiling": False,
    }


def route_context(state: Dict[str, Any]) -> str:
    """Route unknown users to profiling before specialist classification."""
    return "profiling" if state.get("needs_profiling") else "supervisor"