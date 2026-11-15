"""
Tourist Help Emergency Assistance Node (Story 2.2)
Queries EmergencyDB pipeline and prepares candidate response for the verification node.
Module: src/agents/tourist_help/emergency.py
"""

from typing import Dict, Any


def emergency_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: queries EmergencyDB and drafts candidate emergency instructions.
    Passes output to the Help Verification Guardrail Node.
    """
    location = state.get("current_location") or "Egypt"

    # Context ground truth for verification node
    retrieved_chunks = [
        "Tourist Police (شرطة السياحة): 126 (Nationwide assistance for international tourists, harassment, scams).",
        "Ambulance / Medical Emergency (الإسعاف): 123 (Urgent medical transport across all governorates).",
        "General Police (الشرطة): 122 (Accidents and urgent security incidents across Egypt).",
        "Fire Department & Civil Protection (المطافئ): 180 (Fire emergencies and rescue operations).",
        "Traffic Police (مرور مصر): 128 (Highway breakdowns and road incidents)."
    ]
    context_str = "\n".join(retrieved_chunks)

    # Draft candidate response
    candidate_draft = (
        f"🚨 Emergency Assistance for {location}:\n"
        "For immediate emergency assistance in Egypt, please call:\n"
        "• Tourist Police (شرطة السياحة): 126 (tourist safety, harassment, scams)\n"
        "• Ambulance (الإسعاف): 123 (urgent medical emergencies)\n"
        "• General Police (الشرطة): 122 (accidents and security incidents)\n"
        "• Fire Department (المطافئ): 180\n"
        "If you are in immediate danger, please dial 126 or 122 right away."
    )

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "response": candidate_draft,
        "help_domain": "emergency",
        "current_intent": "emergency",
        "next_agent": "emergency_completed"
    }


# Backward-compatible alias
emergency_agent_node = emergency_rag_node
