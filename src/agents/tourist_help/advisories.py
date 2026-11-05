"""
Tourist Help Advisories & Visa Rules Node (Story 2.3 / Dev 2 Stub)
Queries TouristRAG vector store and prepares candidate response for the verification node.
"""

from typing import Dict, Any


def advisories_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: queries TouristRAG for visa and advisory rules.
    Passes output to Dev 4's Help Verification Node.
    """
    user_query = state.get("user_query", "Visa inquiry")

    # Ingest context from TouristRAG (visa rules & entry requirements)
    retrieved_chunks = [
        "Egypt e-Visa: Official electronic visa issued to citizens of listed countries (US, EU, UK, Canada, Japan, etc.).",
        "Fees: Single-entry tourism e-Visa is USD 25-30. Multiple-entry is USD 60-65.",
        "Passport validity: Must be valid for at least 6 months after arrival date.",
        "Nationalities not listed require pre-arranged consular visas prior to travel."
    ]
    context_str = "\n".join(retrieved_chunks)

    candidate_draft = (
        "Egypt provides an official e-Visa for eligible nationalities (including US, EU, UK, Canada, and Japan). "
        "A single-entry e-Visa costs USD 25-30, and your passport must be valid for at least 6 months beyond arrival. "
        "Unlisted nationalities must obtain a consular visa before departure."
    )

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "help_domain": "advisories"
    }
