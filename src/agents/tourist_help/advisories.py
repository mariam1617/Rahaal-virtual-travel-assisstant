"""
Tourist Help Advisories & Visa Rules Node (Story 2.3)
Queries TouristRAG vector store and prepares candidate response for the verification node.
Module: src/agents/tourist_help/advisories.py
"""

from typing import Dict, Any


def advisories_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: queries TouristRAG for visa and advisory rules.
    Passes output to the Help Verification Guardrail Node.
    """
    # TODO (Problem 8): Tourist Help has no real retrieval.
    # Advisories hardcodes both the 'retrieved chunks' and the 'candidate answer'.
    # The user's question is never used (e.g. nationality-specific visa rules or transit questions),
    # so every question gets the identical hardcoded response.
    # Wire to actual vector store or knowledge base retriever querying user_preferences.nationality and user_query.
    # Context ground truth for verification node
    retrieved_chunks = [
        "Egypt e-Visa: Official electronic visa issued to citizens of listed countries (including US, EU, UK, Canada, and Japan).",
        "Fees: Single-entry tourism e-Visa is USD 25-30. Multiple-entry is USD 60-65.",
        "Passport validity: Must be valid for at least 6 months after arrival date.",
        "Eligible passport holders can also obtain a single-entry Visa on Arrival ($25 USD) at major international airports.",
        "Transport & Customs: Official Cairo Metro and GoBus connect major destinations. Declare amounts exceeding $10,000 USD."
    ]
    context_str = "\n".join(retrieved_chunks)

    candidate_draft = (
        "ℹ️ Egypt Travel Advisory & Regulations:\n"
        "• Visas: Egypt provides an official e-Visa for eligible nationalities (including US, EU, UK, Canada, and Japan). "
        "Eligible passport holders can obtain an eVisa online or a Visa on Arrival ($25 USD single-entry).\n"
        "• Passport Validity: Your passport must be valid for at least 6 months beyond arrival.\n"
        "• Transport: Cairo Metro and official GoBus services connect major destinations across Egypt."
    )

    return {
        **state,
        "retrieved_context": context_str,
        "candidate_response": candidate_draft,
        "response": candidate_draft,
        "help_domain": "advisories",
        "current_intent": "advisories",
        "next_agent": "advisories_completed"
    }


# Backward-compatible alias
advisories_agent_node = advisories_rag_node
