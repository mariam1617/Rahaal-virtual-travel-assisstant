"""
Feedback Router Engine (Story 5.2, Story 5.3)
Acts as the classification entry point for any user feedback or changes
requested after an itinerary is generated or booked. It decides whether the
input is a trip adjustment, a long-term preference, or a compound request of both.
Module: src/agents/feedback/router.py
"""

import os
import re
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.state.schema import FeedbackClassification, FeedbackType


FEEDBACK_ROUTER_SYSTEM_PROMPT = """You are the Feedback Router for "Ra7al" (رحال), the Egypt AI Travel Concierge.
Your mission is to classify user feedback or change requests into one of three primary categories:

1. "trip_adjustment":
   - Requests to modify, replace, remove, add, or reschedule activities/places in the current trip plan.
   - Examples: "Change dinner to an Italian restaurant", "Remove the museum on day 2", "Can we do a felucca ride on day 1?", "Make the morning tour start at 10 AM".

2. "long_term_preference":
   - General personal tastes, dietary rules, habits, or style preferences intended to be remembered for future trips.
   - Examples: "I prefer morning flights", "I am vegan", "I always prefer luxury 5-star hotels", "I love historical sites and hate crowded places", "I require wheelchair accessibility".

3. "both":
   - A compound request where the user expresses a personal preference that directly motivates an immediate change to the current itinerary.
   - Examples: "I am vegetarian, so please change tonight's dinner to a vegetarian restaurant", "I prefer morning flights, can we reschedule tomorrow's flight to 8 AM?".

4. "unknown":
   - Vague, ambiguous, or irrelevant feedback with insufficient detail to determine action.

Output your decision strictly matching the FeedbackClassification schema.
"""

# TODO (Problem 17): CONFIDENCE_THRESHOLD is defined here but never actually used in feedback_router_node.
# Apply confidence gating: if decision.confidence < CONFIDENCE_THRESHOLD, route to fallback or ask clarifying question.
CONFIDENCE_THRESHOLD = float(os.getenv("FEEDBACK_ROUTER_THRESHOLD", "0.65"))


def get_feedback_router_llm():
    """
    Instantiate chat model for structured output if keys are present.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    default_model = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

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


def heuristic_feedback_classifier(user_text: str) -> FeedbackClassification:
    """
    Deterministic rule-based backup classifier for zero-network environments and test benchmarks.
    """
    text = (user_text or "").lower().strip()
    if not text:
        return FeedbackClassification(
            feedback_type="unknown",
            confidence=0.0,
            reasoning="Empty input text."
        )

    # 1. Signals for active trip adjustment
    trip_change_keywords = [
        "change", "modify", "remove", "replace", "swap", "add", "cancel",
        "reschedule", "delete", "substitute", "switch", "earlier", "later",
        "instead", "drop", "update", "move"
    ]
    has_trip_keyword = any(re.search(rf"\b{kw}\b", text) for kw in trip_change_keywords)

    trip_context_keywords = [
        "day 1", "day 2", "day 3", "day 4", "day 5", "day 6", "day 7",
        "dinner", "lunch", "breakfast", "morning", "afternoon", "evening",
        "tonight", "tomorrow", "itinerary", "plan", "tour", "hotel", "museum", "activity"
    ]
    has_trip_context = any(kw in text for kw in trip_context_keywords)

    is_trip_adjustment = has_trip_keyword and (has_trip_context or len(text.split()) > 2)

    # 2. Signals for long-term preference
    preference_keywords = [
        "i prefer", "i always", "i never", "i like", "i dislike", "i hate",
        "i love", "my preference", "habit", "i am vegan", "i'm vegan",
        "i am vegetarian", "i'm vegetarian", "i eat halal", "halal only",
        "kosher", "gluten-free", "morning flights", "evening flights",
        "window seat", "aisle seat", "direct flights", "luxury hotels",
        "boutique hotels", "budget traveler", "wheelchair", "accessibility",
        "keep in mind that i", "for future trips", "always book"
    ]
    is_long_term_preference = any(kw in text for kw in preference_keywords)

    # 3. Compound check (Both)
    if is_trip_adjustment and is_long_term_preference:
        return FeedbackClassification(
            feedback_type="both",
            confidence=0.92,
            reasoning="Query expresses both a personal travel preference and a trip modification request.",
            modification_intent=text,
            preference_intent=text
        )

    # 4. Standalone Trip Adjustment
    if is_trip_adjustment:
        return FeedbackClassification(
            feedback_type="trip_adjustment",
            confidence=0.90,
            reasoning="User requested a specific adjustment to the active itinerary.",
            modification_intent=text
        )

    # 5. Standalone Long-Term Preference
    if is_long_term_preference:
        return FeedbackClassification(
            feedback_type="long_term_preference",
            confidence=0.88,
            reasoning="User expressed a personal preference, habit, or dietary restriction.",
            preference_intent=text
        )

    # 6. Fallback / Ambiguous
    return FeedbackClassification(
        feedback_type="unknown",
        confidence=0.40,
        reasoning="Ambiguous input without clear trip adjustment or preference keywords."
    )


def extract_user_prompt_from_state(state: Dict[str, Any]) -> str:
    """Extracts the latest user input string from various state formats."""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, tuple):
            return str(last_msg[1])
        if hasattr(last_msg, "content"):
            return str(last_msg.content)
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
        return str(last_msg)
    return str(state.get("user_query") or "")


def feedback_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Classifies incoming feedback from active chat and directs
    flow to Active Itinerary Modifier, Insight Extractor, or Fallback.
    """
    user_prompt = extract_user_prompt_from_state(state)
    llm = get_feedback_router_llm()
    decision: Optional[FeedbackClassification] = None

    if llm:
        try:
            structured_llm = llm.with_structured_output(FeedbackClassification)
            system_msg = SystemMessage(content=FEEDBACK_ROUTER_SYSTEM_PROMPT)
            user_msg = HumanMessage(content=f"User Feedback: {user_prompt}")
            decision = structured_llm.invoke([system_msg, user_msg])
        except Exception as e:
            decision = heuristic_feedback_classifier(user_prompt)
    else:
        decision = heuristic_feedback_classifier(user_prompt)

    # TODO (Problem 17): Enforce CONFIDENCE_THRESHOLD here. If decision.confidence < CONFIDENCE_THRESHOLD,
    # route to feedback_fallback or profiling rather than blindly executing low-confidence decisions.
    # Map decision to next_agent
    if decision.feedback_type == "trip_adjustment":
        next_agent = "itinerary_modifier"
    elif decision.feedback_type in ["long_term_preference", "both"]:
        next_agent = "insight_extractor"
    else:
        next_agent = "feedback_fallback"

    return {
        "feedback_type": decision.feedback_type,
        "feedback_confidence": decision.confidence,
        "feedback_reasoning": decision.reasoning,
        "feedback_classification": decision.model_dump(),
        "next_agent": next_agent
    }


def route_feedback(state: Dict[str, Any]) -> str:
    """
    LangGraph Conditional Edge router mapping feedback classification to downstream nodes.
    """
    fb_type = state.get("feedback_type", "unknown")
    if fb_type == "trip_adjustment":
        return "itinerary_modifier"
    if fb_type in ["long_term_preference", "both"]:
        return "insight_extractor"
    return "feedback_fallback"
