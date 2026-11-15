"""
Tourist Help Emergency Assistance Node (Story 2.2 / Dev 3 Stub)
Queries EmergencyDB pipeline and prepares candidate response for the verification node.
Tourist Help Subgraph - Emergency Assistance Agent Node (Story 2.2)
Module: src/agents/tourist_help/emergency.py
"""

from typing import Dict, Any


def emergency_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: queries EmergencyDB and drafts candidate emergency instructions.
    Passes output to Dev 4's Help Verification Node.
    """
    user_query = state.get("user_query", "Emergency help")

    # Ingest context from EmergencyDB or default registry
    retrieved_chunks = [
        "Tourist Police (شرطة السياحة): 126 (Nationwide assistance for international tourists, harassment, scams).",
        "Ambulance / Medical Emergency (الإسعاف): 123 (Urgent medical transport across all governorates).",
        "General Police (الشرطة): 122 (Accidents and urgent security incidents)."
    ]
    context_str = "\n".join(retrieved_chunks)

    # Draft candidate response
    candidate_draft = (
        "For immediate emergency assistance in Egypt, please call:\n"
        "- Tourist Police: 126 (tourist safety and scams)\n"
        "- Ambulance: 123 (medical emergencies)\n"
        "- General Police: 122 (accidents and security)"
def emergency_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emergency Assistance Agent node.
    Handles critical distress, police, medical, and emergency queries.
    Provides immediate contact information (e.g. Tourist Police 126, Police 122, Ambulance 123).
    """
    location = state.get("current_location", "Egypt")
    
    response_text = (
        f" Emergency Assistance for {location}:\n"
        "• Tourist Police (شرطة السياحة): 126\n"
        "• General Police (الشرطة): 122\n"
        "• Ambulance (الإسعاف): 123\n"
        "• Fire Department (المطافئ): 180\n"
        "If you are in immediate danger, please dial the numbers above right away."
    )

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "help_domain": "emergency"
   
        "response": response_text,
        "next_agent": "emergency_completed"
    }

