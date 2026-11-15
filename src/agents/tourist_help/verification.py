"""
Tourist Help Verification Guardrail Node 
Provides an LLM verification step acting as a guardrail against hallucinations,
invalid emergency numbers, and ungrounded travel advisories.
"""

import os
import re
from typing import Dict, Any, Optional, List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agents.tourist_help.schemas import (
    RAGInputContext,
    VerificationResult,
    HelpDomain
)
from src.agents.tourist_help.prompts import (
    HELP_VERIFICATION_SYSTEM_PROMPT,
    OFFICIAL_EGYPT_EMERGENCY_NUMBERS,
    DEFAULT_EMERGENCY_FALLBACK,
    DEFAULT_ADVISORY_FALLBACK,
    DEFAULT_GENERAL_FALLBACK,
    format_verification_user_prompt
)

# Known foreign emergency numbers that must never be presented for Egypt
FOREIGN_EMERGENCY_NUMBERS = [r"\b911\b", r"\b999\b", r"\b112\b"]
FOREIGN_EMERGENCY_REGEX = re.compile(r"|".join(FOREIGN_EMERGENCY_NUMBERS))


# =========================================================================
# LLM Factory for Verification Guardrail
# =========================================================================

def get_verification_llm():
    """
    Instantiate chat model for structured output verification.
    Supports Google Gemini (via langchain-google-genai) and OpenAI.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    default_model = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")

    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=default_model,
            google_api_key=gemini_key,
            temperature=0.0
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.0
        )
    return None


# =========================================================================
# Fast Deterministic Checks & Heuristic Fallback
# =========================================================================

def detect_foreign_emergency_numbers(text: str) -> List[str]:
    """Scan candidate text for foreign emergency numbers (911, 999, 112)."""
    matches = FOREIGN_EMERGENCY_REGEX.findall(text)
    return list(set(matches))


def extract_phone_numbers(text: str) -> List[str]:
    """Extract standard telephone numbers, hotlines, and short codes mentioned in text."""
    # Matches digits between 3 and 12 characters (short codes, hotlines, landlines, mobile numbers)
    candidates = re.findall(r"\b\d{3,12}\b", text)
    # Exclude common 4-digit calendar years (2020-2035) so dates are not mistaken for phone numbers
    return [c for c in candidates if not (len(c) == 4 and (2020 <= int(c) <= 2035))]


def heuristic_verification_fallback(input_data: RAGInputContext) -> VerificationResult:
    """
    Deterministic rule-based verification fallback.
    Used for instant rejection of critical errors or when LLM API is unavailable/offline.
    """
    candidate = input_data.candidate_response
    context = input_data.retrieved_context or ""
    domain = input_data.domain

    # 1. Hard Guard: Check for foreign emergency numbers
    foreign_matches = detect_foreign_emergency_numbers(candidate)
    if foreign_matches:
        return VerificationResult(
            is_grounded=False,
            has_hallucination=True,
            critical_entities_valid=False,
            confidence_score=1.0,
            hallucination_details=[
                f"Foreign emergency number(s) detected: {', '.join(foreign_matches)}. Invalid for Egypt."
            ],
            reasoning="Candidate response recommends non-Egyptian emergency number(s) that are ineffective in Egypt.",
            safe_response=DEFAULT_EMERGENCY_FALLBACK
        )

    # 2. Empty Context Guard: If context is empty and candidate asserts specific facts
    if not context.strip():
        # If candidate gives generic advice without specific ungrounded facts, pass with caution; otherwise fallback
        candidate_numbers = extract_phone_numbers(candidate)
        if candidate_numbers:
            # Check if all numbers are in official Egypt emergency directory
            known_valid_numbers = set(OFFICIAL_EGYPT_EMERGENCY_NUMBERS.values())
            if all(num in known_valid_numbers for num in candidate_numbers):
                return VerificationResult(
                    is_grounded=True,
                    has_hallucination=False,
                    critical_entities_valid=True,
                    confidence_score=0.9,
                    hallucination_details=[],
                    reasoning="Context was empty, but candidate numbers match official Egypt directory exactly.",
                    safe_response=candidate
                )
            else:
                fallback_msg = DEFAULT_EMERGENCY_FALLBACK if domain == "emergency" else DEFAULT_GENERAL_FALLBACK
                return VerificationResult(
                    is_grounded=False,
                    has_hallucination=True,
                    critical_entities_valid=False,
                    confidence_score=0.85,
                    hallucination_details=["Unknown phone numbers provided without supporting context."],
                    reasoning="Context was empty and candidate provided unverified phone numbers.",
                    safe_response=fallback_msg
                )

    # 3. Emergency Domain Validation: Ensure numbers in candidate exist in context or official registry
    if domain == "emergency":
        candidate_numbers = extract_phone_numbers(candidate)
        known_valid_numbers = set(OFFICIAL_EGYPT_EMERGENCY_NUMBERS.values())
        invalid_numbers = [num for num in candidate_numbers if (num not in context and num not in known_valid_numbers)]

        if invalid_numbers:
            return VerificationResult(
                is_grounded=False,
                has_hallucination=True,
                critical_entities_valid=False,
                confidence_score=0.95,
                hallucination_details=[f"Unverified emergency number(s): {', '.join(invalid_numbers)}"],
                reasoning="Response contains emergency numbers not found in retrieved context or Egypt directory.",
                safe_response=DEFAULT_EMERGENCY_FALLBACK
            )

    # 4. Visa & Advisory Domain Validation: Ensure key terms are grounded
    if domain in ("visa", "advisories"):
        # Flag if candidate promises visa on arrival / visa free when context does not mention them
        if "visa on arrival" in candidate.lower() and "visa on arrival" not in context.lower():
            return VerificationResult(
                is_grounded=False,
                has_hallucination=True,
                critical_entities_valid=False,
                confidence_score=0.9,
                hallucination_details=["Candidate promised 'visa on arrival' which does not appear in retrieved context."],
                reasoning="Visa on arrival promised without retrieved context substantiation.",
                safe_response=DEFAULT_ADVISORY_FALLBACK
            )

    # Default Pass: Grounded
    return VerificationResult(
        is_grounded=True,
        has_hallucination=False,
        critical_entities_valid=True,
        confidence_score=0.9,
        hallucination_details=[],
        reasoning="Heuristic check verified candidate response against context and official Egyptian registry.",
        safe_response=candidate
    )


# =========================================================================
# Core Verification Function
# =========================================================================

def verify_rag_response(
    input_data: RAGInputContext,
    llm: Optional[Any] = None
) -> VerificationResult:
    """
    Evaluates candidate response against retrieved context.
    Executes fast deterministic pre-checks first, then queries the LLM judge
    with structured output, falling back to heuristic verification if needed.
    """
    # Fast Path 1: Check for dangerous foreign emergency numbers immediately
    foreign_matches = detect_foreign_emergency_numbers(input_data.candidate_response)
    if foreign_matches:
        return VerificationResult(
            is_grounded=False,
            has_hallucination=True,
            critical_entities_valid=False,
            confidence_score=1.0,
            hallucination_details=[
                f"Candidate recommended foreign emergency number(s): {', '.join(foreign_matches)}."
            ],
            reasoning="Fast-path guardrail intercepted non-Egyptian emergency number.",
            safe_response=DEFAULT_EMERGENCY_FALLBACK
        )

    # Resolve LLM
    active_llm = llm or get_verification_llm()

    if active_llm:
        try:
            structured_judge = active_llm.with_structured_output(VerificationResult)
            system_msg = SystemMessage(content=HELP_VERIFICATION_SYSTEM_PROMPT)
            user_msg = HumanMessage(
                content=format_verification_user_prompt(
                    user_query=input_data.user_query,
                    retrieved_context=input_data.retrieved_context,
                    candidate_response=input_data.candidate_response,
                    domain=input_data.domain
                )
            )
            result: VerificationResult = structured_judge.invoke([system_msg, user_msg])

            # Safety fallback post-processing: If rejected but safe_response was unaltered
            if (not result.is_grounded or result.has_hallucination) and (
                result.safe_response.strip() == input_data.candidate_response.strip()
            ):
                if input_data.domain == "emergency":
                    result.safe_response = DEFAULT_EMERGENCY_FALLBACK
                elif input_data.domain in ("visa", "advisories"):
                    result.safe_response = DEFAULT_ADVISORY_FALLBACK
                else:
                    result.safe_response = DEFAULT_GENERAL_FALLBACK

            return result

        except Exception as e:
            print(f"[HelpVerification] LLM verification failed ({e}), using heuristic fallback.")
            return heuristic_verification_fallback(input_data)

    # If no LLM available, use deterministic heuristic verifier
    return heuristic_verification_fallback(input_data)


# =========================================================================
# LangGraph Node Implementation
# =========================================================================

def _normalize_context(raw_context: Any) -> str:
    """Helper to convert documents, lists of strings/dicts, or text into a single context string."""
    if not raw_context:
        return ""
    if isinstance(raw_context, str):
        return raw_context
    if isinstance(raw_context, list):
        chunks = []
        for item in raw_context:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("page_content") or str(item)
                chunks.append(content)
            elif hasattr(item, "page_content"):
                chunks.append(item.page_content)
            else:
                chunks.append(str(item))
        return "\n\n".join(chunks)
    return str(raw_context)


def help_verification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Help Response Verification.
    Validates RAG outputs from EmergencyDB and TouristRAG before presentation.
    """
    # 1. Extract User Query
    user_query = state.get("user_query")
    if not user_query:
        messages = state.get("messages", [])
        if messages:
            last_human = next(
                (m for m in reversed(messages) if getattr(m, "type", None) == "human" or (isinstance(m, tuple) and m[0] == "user")),
                messages[-1]
            )
            user_query = getattr(last_human, "content", None) or (last_human[1] if isinstance(last_human, tuple) else str(last_human))
        else:
            user_query = "Help request"

    # 2. Extract Retrieved Context
    raw_context = state.get("retrieved_context") or state.get("context") or state.get("rag_docs") or ""
    retrieved_context = _normalize_context(raw_context)

    # 3. Extract Candidate Response
    candidate_response = (
        state.get("candidate_response")
        or state.get("draft_response")
        or state.get("response")
    )
    if not candidate_response:
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            candidate_response = getattr(last_msg, "content", None) or (last_msg[1] if isinstance(last_msg, tuple) else str(last_msg))
        else:
            candidate_response = DEFAULT_GENERAL_FALLBACK

    # 4. Resolve Domain
    domain: HelpDomain = state.get("help_domain") or state.get("intent_category") or "general"
    if domain not in ("emergency", "advisories", "visa", "transit", "general"):
        domain = "general"

    # 5. Build input payload & execute verification
    input_payload = RAGInputContext(
        user_query=str(user_query),
        retrieved_context=retrieved_context,
        candidate_response=str(candidate_response),
        domain=domain
    )
    result = verify_rag_response(input_payload)

    # 6. Prepare State Updates
    output_messages = list(state.get("messages", []))
    verified_ai_message = AIMessage(content=result.safe_response)
    output_messages.append(verified_ai_message)

    errors = list(state.get("errors", []))
    if result.has_hallucination or not result.is_grounded:
        errors.append(f"Guardrail intercepted response: {result.reasoning}")

    return {
        **state,
        "messages": output_messages,
        "response": result.safe_response,
        "verified_response": result.safe_response,
        "is_verified": result.is_grounded and not result.has_hallucination,
        "verification_result": result.model_dump(),
        "errors": errors,
        "next_agent": "completed"
    }
