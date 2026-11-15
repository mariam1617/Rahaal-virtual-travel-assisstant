"""
Tourist Help Subgraph Package (Story 2.1 - 2.4)
"""

from src.agents.tourist_help.schemas import (
    RAGInputContext,
    VerificationResult,
    HelpDomain,
)
from src.agents.tourist_help.prompts import (
    HELP_VERIFICATION_SYSTEM_PROMPT,
    OFFICIAL_EGYPT_EMERGENCY_NUMBERS,
    DEFAULT_EMERGENCY_FALLBACK,
    DEFAULT_ADVISORY_FALLBACK,
    DEFAULT_GENERAL_FALLBACK,
)
from src.agents.tourist_help.verification import (
    help_verification_node,
    verify_rag_response,
    detect_foreign_emergency_numbers,
    heuristic_verification_fallback,
    get_verification_llm,
)
from src.agents.tourist_help.intent_resolver import (
    extract_coordinates,
    resolve_location_node,
    parse_tourist_intent,
    resolve_tourist_help_intent,
    intent_parser_node,
    intent_resolver_node,
    route_request,
    route_help_subintent,
    fallback_agent_node,
)
from src.agents.tourist_help.emergency import (
    emergency_rag_node,
    emergency_agent_node,
)
from src.agents.tourist_help.advisories import (
    advisories_rag_node,
    advisories_agent_node,
)
from src.agents.tourist_help.recommendations import (
    recommendations_node,
    recommendations_agent_node,
)
from src.agents.tourist_help.graph import (
    create_tourist_help_graph,
    tourist_help_app,
)

__all__ = [
    # Schemas & Prompts
    "RAGInputContext",
    "VerificationResult",
    "HelpDomain",
    "HELP_VERIFICATION_SYSTEM_PROMPT",
    "OFFICIAL_EGYPT_EMERGENCY_NUMBERS",
    "DEFAULT_EMERGENCY_FALLBACK",
    "DEFAULT_ADVISORY_FALLBACK",
    "DEFAULT_GENERAL_FALLBACK",
    # Verification Guardrail
    "help_verification_node",
    "verify_rag_response",
    "detect_foreign_emergency_numbers",
    "heuristic_verification_fallback",
    "get_verification_llm",
    # Location & Intent Resolution
    "extract_coordinates",
    "resolve_location_node",
    "parse_tourist_intent",
    "resolve_tourist_help_intent",
    "intent_parser_node",
    "intent_resolver_node",
    "route_request",
    "route_help_subintent",
    "fallback_agent_node",
    # RAG Nodes
    "emergency_rag_node",
    "emergency_agent_node",
    "advisories_rag_node",
    "advisories_agent_node",
    "recommendations_node",
    "recommendations_agent_node",
    # Graph App
    "create_tourist_help_graph",
    "tourist_help_app",
]
