"""
Tourist Help Subgraph - Contextual Place Recommendations Agent Node (Story 2.4)
Module: src/agents/tourist_help/recommendations.py
"""

from typing import Dict, Any


def recommendations_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Contextual Place Recommendations Agent node.
    Suggests nearby dining, cultural landmarks, and attractions based on current location and preferences.
    """
    location = state.get("current_location", "Cairo, Egypt")
    
    response_text = (
        f"📍 Local Recommendations near {location}:\n"
        "• Top Dining: Try authentic Koshary Abou Tarek or local street cafes.\n"
        "• Cultural Highlights: Visit the nearby historical quarters and bazaars (Khan el-Khalili).\n"
        "• Best Times: Late afternoon and evening offer the best walking atmosphere."
    )

    return {
        **state,
        "response": response_text,
        "next_agent": "recommendations_completed"
    }

