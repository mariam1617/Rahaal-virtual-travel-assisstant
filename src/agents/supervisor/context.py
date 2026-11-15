"""Context hydration for Supervisor requests (Story 1.1)."""

from typing import Any, Dict

from src.state.memory import MemoryLoader


def hydrate_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load known user preferences and preserve request/session context."""
    user_id = state.get("user_id")
    if not user_id:
        return {**state, "context_loaded": False}

    profile = MemoryLoader.get_user_profile(user_id)
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