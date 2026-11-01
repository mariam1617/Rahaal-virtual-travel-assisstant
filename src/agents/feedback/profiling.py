

from typing import Dict, Any
from langchain_core.messages import AIMessage

PROFILING_GREETING_PROMPT = (
    "Ahlan! Welcome to Ra7al (رحال), your personal AI travel concierge for Egypt.\n\n"
)


def profiling_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profiling & Interview Agent stub/passthrough.
    Triggered when user intent is ambiguous or low-confidence (< 0.65).
    """
    ai_msg = AIMessage(content=PROFILING_GREETING_PROMPT)
    return {
        **state,
        "messages": state.get("messages", []) + [ai_msg],
        "response": PROFILING_GREETING_PROMPT,
        "next_agent": "profiling_completed"
    }

