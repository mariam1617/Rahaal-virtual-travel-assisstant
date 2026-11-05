"""
Tourist Help Recommendations Node (Story 2.4 / Dev Stub)
Provides local attractions and food recommendations filtered by preferences.
"""

from typing import Dict, Any


def recommendations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: recommends nearby attractions and places in Egypt.
    Passes output to Dev 4's Help Verification Node.
    """
    candidate_draft = (
        "Here are highly recommended authentic spots in Cairo:\n"
        "- Koshary Abou Tarek (Downtown Cairo) for authentic national dish.\n"
        "- Khan el-Khalili historic bazaar and El Fishawy Cafe for traditional mint tea.\n"
        "- The Grand Egyptian Museum (GEM) in Giza for world-class antiquities."
    )
    context_str = "Cairo cultural landmarks and culinary recommendations: Abou Tarek, Khan el-Khalili, GEM."

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "help_domain": "recommendations"
    }
