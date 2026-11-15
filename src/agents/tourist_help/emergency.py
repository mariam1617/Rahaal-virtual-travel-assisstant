"""
Tourist Help Subgraph - Emergency Assistance Agent Node (Story 2.2)
Module: src/agents/tourist_help/emergency.py
"""

from typing import Dict, Any


def emergency_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emergency Assistance Agent node.
    Handles critical distress, police, medical, and emergency queries.
    Provides immediate contact information (e.g. Tourist Police 126, Police 122, Ambulance 123).
    """
    location = state.get("current_location", "Egypt")
    
    response_text = (
        f"🚨 Emergency Assistance for {location}:\n"
        "• Tourist Police (شرطة السياحة): 126\n"
        "• General Police (الشرطة): 122\n"
        "• Ambulance (الإسعاف): 123\n"
        "• Fire Department (المطافئ): 180\n"
        "If you are in immediate danger, please dial the numbers above right away."
    )

    return {
        **state,
        "response": response_text,
        "next_agent": "emergency_completed"
    }

