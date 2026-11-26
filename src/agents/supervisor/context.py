"""Context hydration for Supervisor requests (Story 1.1)."""
#UPDATE 21/11: 
# Added SupervisorState as a TypedDict .

from typing import Any, Dict
from src.state.memory import MemoryLoader
from src.state.app_state import AppState


def hydrate_context_node(state: AppState) -> Dict[str, Any]:
    """Load known user preferences and preserve request/session context."""
    # TODO (Problem 13): All users share one profile if user_id falls back to a common test_user.
    # Ensure guest/anonymous sessions do not overwrite each other or share persistent profiles.
    user_id = state.get("user_id")
    if not user_id:
        return {"context_loaded": False}

    try:
        # TODO (Problem 2): Profile logic exists three times (ProfileStore, MemoryLoader, SQLiteUserRepository).
        # Collapse into a unified ProfileService on top of SQLiteUserRepository with separate get and ensure methods.
        profile = MemoryLoader.get_user_profile(user_id)
    except Exception:
        return {
            "context_loaded": False,
            "context_warning": (
                f"Unable to load profile for user '{user_id}' due to a "
                "database or connection error."
            ),
            "needs_profiling": True,
            "next_agent": "profiling",
        }

    if profile is None:
        # Only force profiling if the user sent a vague/short query.
        # Detailed queries (plan trips, visa questions, etc.) should go straight to supervisor.
        messages = state.get("messages", [])
        last_text = ""
        if messages:
            last = messages[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                last_text = str(last[1])
            elif hasattr(last, "content"):
                last_text = str(last.content)

        force_profiling = len(last_text.strip()) < 15  # only short/vague queries need profiling
        return {
            "context_loaded": False,
            "context_warning": f"No profile found for user '{user_id}'.",
            "needs_profiling": force_profiling,
            "next_agent": "profiling" if force_profiling else "",
        }

    return {
        "user_preferences": profile,
        "context_loaded": True,
        "needs_profiling": False,
    }

def route_context(state: Dict[str, Any]) -> str:
    """Route unknown users to profiling before specialist classification.
    
    Only force profiling when the user genuinely has no profile in the DB.
    If context failed to load due to a transient error (context_loaded=False but no
    needs_profiling flag), still let the supervisor classify the intent so the user's
    actual request is handled.
    """
    return "profiling" if state.get("needs_profiling") else "supervisor"