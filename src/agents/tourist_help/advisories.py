"""
Tourist Help Subgraph - Advisories & Rules Agent Node (Story 2.3)
Module: src/agents/tourist_help/advisories.py
"""

from typing import Dict, Any


def advisories_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    General Tourist Advisories & Rules Agent node.
    Handles inquiries about Egyptian visas, entry requirements, customs, and transit rules.
    """
    response_text = (
        "ℹ️ Egypt Travel Advisory & Regulations:\n"
        "• Visas: Eligible passport holders can obtain an eVisa or Visa on Arrival ($25 USD single-entry).\n"
        "• Currency & Customs: Declare amounts exceeding $10,000 USD upon arrival.\n"
        "• Transport: Cairo Metro and official GoBus services connect major destinations."
    )

    return {
        **state,
        "response": response_text,
        "next_agent": "advisories_completed"
    }

