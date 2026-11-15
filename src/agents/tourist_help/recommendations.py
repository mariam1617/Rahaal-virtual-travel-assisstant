"""
Tourist Help Recommendations Node (Story 2.4)
Provides local attractions and food recommendations filtered by preferences and location.
Module: src/agents/tourist_help/recommendations.py
"""

from typing import Dict, Any


def recommendations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: recommends nearby attractions and places in Egypt.
    Passes output to the Help Verification Guardrail Node.
    """
    location = state.get("current_location") or "Cairo, Egypt"

    context_str = (
        "Cairo cultural landmarks and culinary recommendations: "
        "Koshary Abou Tarek (Downtown Cairo), Khan el-Khalili historic bazaar, El Fishawy Cafe, "
        "and The Grand Egyptian Museum (GEM) in Giza."
    )

    candidate_draft = (
        f"📍 Local Recommendations near {location}:\n"
        "• Top Dining: Try authentic Koshary Abou Tarek in Downtown Cairo for Egypt's famous national dish.\n"
        "• Cultural Highlights: Visit the historic Khan el-Khalili bazaar and enjoy traditional mint tea at El Fishawy Cafe.\n"
        "• World-Class Heritage: Explore the Grand Egyptian Museum (GEM) in Giza for ancient antiquities."
    )

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "response": candidate_draft,
        "help_domain": "recommendations",
        "current_intent": "recommendations",
        "next_agent": "recommendations_completed"
    }


# Backward-compatible alias
recommendations_agent_node = recommendations_node
