"""
Supervisor Request Routing Engine (Story 1.2 / Dev 4)
Item 1: Supervisor Prompt Engineering & Intent Definition
"""
#UPDATE 21/11: 
# Added SupervisorState as a TypedDict .
# Added add_messages for LangGraph-managed message history.
# Changed StateGraph(dict) to StateGraph(SupervisorState).
# Removed manual message concatenation in the supervisor out-of-scope and profiling nodes.
# Updated overall_state.py to use add_messages

from typing import  Any, Dict
from pydantic import  Field
from langchain_core.messages import AIMessage,  HumanMessage, SystemMessage
from src import state
from src.state.app_state import AppState
from src.agents.supervisor.state import SupervisorDecision, TargetSubgraph

SupervisorState = AppState  # DEPRECATED alias




SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Router for "Ra7al" (رحال), an Egypt-exclusive AI Travel Concierge.
Your mission is to accurately parse the user's travel request, strictly enforce geographic boundaries, and route the request to the correct specialized subgraph using the structured output format.

### Geographic Boundary Rules (CRITICAL)
- Ra7al is 100% EXCLUSIVE to Egypt travel (inbound tourists or domestic travelers).
- If the user asks to plan travel, book trips, or get recommendations for ANY country/city outside Egypt (e.g. Paris, Dubai, Tokyo, London, Bali), you MUST classify:
  - is_egypt_scoped = False
  - target = "out_of_scope"
  - confidence = 1.0
  - reasoning = "User requested a destination outside Egypt."

### Domain Classification Criteria
1. "planner":
   - Explicit multi-day itinerary requests (e.g., "Plan a 3-day trip to Cairo", "5 days in Luxor and Aswan").
   - Day-by-day scheduling of sights, temples, and activities with a clear duration or date range.
   - Budget-based travel planning requests.
   - DO NOT use "planner" for simple factual questions about how to travel between cities.

2. "tourist_help":
   - ALL factual traveler information questions, including:
     * Visa/passport/eVisa/entry requirements.
     * Emergency numbers (Tourist Police 126, Ambulance 123, Police 122, Fire 180).
     * Transportation OPTIONS between cities (e.g., "How can I travel from Cairo to Luxor?", "Is there a bus from Cairo to Alexandria?").
     * Taxi, tipping/baksheesh, Uber, Careem information.
     * Safety advisories, crime risks, travel warnings.
     * Hospital, medical, and pharmacy information.
     * Cultural rules: dress codes, alcohol, photography, drones, Ramadan.
     * SIM cards, currency exchange, customs.
     * Local place recommendations, cafes, restaurants.
   - KEY RULE: If the user asks HOW to travel between two Egyptian cities (transport options, bus times, train info), route to "tourist_help" — NOT planner. Only route to planner if they ask for a full trip schedule with days.

3. "execution":
   - Explicit intent to BOOK or RESERVE: flights, buses (GoBus), trains (ENR), hotels.
   - Calendar synchronization requests (e.g., "Sync this schedule to my Google Calendar").
   - Must use the word 'book', 'reserve', 'ticket', or 'calendar'.

4. "feedback":
   - Modifying or tweaking an existing generated itinerary (e.g., "Remove the museum on day 2", "I prefer vegetarian food instead").
   - Submitting reviews, ratings, or feedback about past trip activities.

5. "profiling":
   - Ambiguous, vague, or generic queries that lack clear travel details (e.g., "Hi", "Help me plan something", "I want to travel", "What can you do?").
   - When the user's intent is unclear or confidence is below 0.65, route to profiling.
   - NEVER route to profiling for clear factual questions about Egypt travel (visas, transport, safety, emergency).

6. "out_of_scope":
   - Non-Egypt travel requests or non-travel queries (e.g., coding, math, general chat).

7. "FINISH":
   - Use this only when a previous subgraph completed the request or no more subgraph handoffs are needed.
   - Never repeat a subgraph that appears in completed_steps.

### Examples of Correct Routing
- "How can I travel from Cairo to Luxor by bus?" → tourist_help
- "What transportation options are there from Cairo to Alexandria?" → tourist_help
- "How much should I tip a taxi driver in Cairo?" → tourist_help
- "Is it safe to travel to Luxor?" → tourist_help
- "Where is the nearest hospital in Cairo?" → tourist_help
- "What are the latest safety advisories for tourists in Egypt?" → tourist_help
- "Plan a 3-day trip to Cairo" → planner
- "Book a GoBus ticket from Cairo to Alexandria" → execution
- "I want to travel" → profiling

Output your decision strictly adhering to the SupervisorDecision schema.
"""

OUT_OF_SCOPE_RESPONSE = (
    "Ra7al (رحال) is an Egypt-exclusive travel concierge dedicated to crafting authentic journeys across Egypt. "
    "We currently only plan and assist with travel within Egypt (Cairo, Alexandria, Luxor, Aswan, Red Sea, Sinai, Siwa, etc.).\n\n"
    "Could I help you discover one of Egypt's incredible historical wonders or coastal destinations instead?"
)


# =========================================================================
# Item 2: LLM Factory & Supervisor Router Node
# =========================================================================

import os
import re
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))


@lru_cache(maxsize=4)
def _create_router_llm(gemini_key: Optional[str], openai_key: Optional[str], model_name: str):
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=0.0,
        )
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.0,
        )
    return None


def get_router_llm():
    """
    Instantiate the chat model with structured output capability.
    Supports Google Gemini (via langchain-google-genai) and OpenAI.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("DEFAULT_MODEL", DEFAULT_GEMINI_MODEL)
    return _create_router_llm(gemini_key, openai_key, model_name)


def heuristic_fallback_classifier(user_text: str) -> SupervisorDecision:
    """
    Deterministic backup classifier used if LLM API is unavailable or in offline testing.
    Guarantees robust routing under zero network connectivity.
    """
    # TODO (Problem 9): Substring keyword matching misfires.
    # Check all keyword lists and regex patterns to ensure whole-word boundaries (\b) are strictly used,
    # and that intent classification prevents collisions (e.g. "scam" should route advice queries to
    # advisories, not emergency; "eat" should not match substrings in other words).
    text = user_text.lower().strip()
    
    # 1. Geographic Exclusion (Non-Egypt destinations)
    non_egypt_keywords = ["paris", "dubai", "tokyo", "london", "rome", "new york", "bali", "turkey", "spain", "france", "italy", "greece", "japan", "morocco"]
    for kw in non_egypt_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return SupervisorDecision(
                target="out_of_scope",
                confidence=1.0,
                is_egypt_scoped=False,
                reasoning=f"User requested '{kw}', which is outside Egypt."
            )

    # 2. Feedback / Modifying plan
    feedback_keywords = ["change", "modify", "remove", "don't like", "update", "replace", "rating", "review"]
    if any(re.search(rf"\b{re.escape(fw)}\b", text) for fw in feedback_keywords):
        return SupervisorDecision(
            target="feedback",
            confidence=0.85,
            is_egypt_scoped=True,
            reasoning="User requested changes to an active travel plan or provided feedback."
        )

    # 3. Execution / Booking / Calendar
    # NOTE: "bus to" and "train to" are intentionally removed — those are information
    # queries, not bookings. Only explicit booking verbs trigger execution routing.
    booking_keywords = ["book", "reserve", "ticket", "calendar", "sync", "flight ticket", "gobus", "hotel"]
    if any(re.search(rf"\b{re.escape(bw)}\b", text) for bw in booking_keywords):
        return SupervisorDecision(
            target="execution",
            confidence=0.90,
            is_egypt_scoped=True,
            reasoning="User explicitly requested booking or calendar synchronization."
        )

    # 4. Tourist Help — factual traveler information (checked BEFORE planner to prevent
    #    intercity transport questions being misclassified as itinerary requests)
    help_keywords = [
        # Visa & entry
        "visa", "evisa", "e-visa", "passport", "immigration", "embassy",
        # Emergency & medical
        "emergency", "police", "ambulance", "hospital", "doctor", "clinic", "pharmacy",
        # Transport info (how to travel between cities — NOT booking)
        "transport", "transportation", "how to travel", "how can i travel",
        "how do i get", "how to get", "getting to", "getting from",
        "taxi", "taxis", "metro", "uber", "careem", "ferry",
        # Tipping & money
        "tipping", "tip", "baksheesh",
        # Safety
        "safety", "safe", "dangerous", "danger", "advisory", "advisories",
        "scam", "scams", "crime",
        # Cultural rules
        "dress code", "sim card", "currency", "alcohol", "photography",
        # Recommendations
        "recommend", "koshary", "near me", "nearby", "cafes", "restaurant",
    ]
    if any(re.search(rf"\b{re.escape(hw)}\b", text) for hw in help_keywords):
        return SupervisorDecision(
            target="tourist_help",
            confidence=0.88,
            is_egypt_scoped=True,
            reasoning="User requested factual tourist advice, visa information, transport options, emergency help, or nearby POIs."
        )
    # Also catch multi-word transport patterns not covered by single-word keywords
    transport_patterns = [
        r"how\s+(?:can|do|could|would)\s+(?:i|we)\s+(?:travel|get|go|reach)",
        r"(?:travel|get)\s+from\s+\w+\s+to\s+\w+",
        r"(?:is\s+there\s+a?\s*|what\s+(?:are\s+the?\s*)?)?(?:bus|train|flight)\s+(?:from|between|to)",
        r"how\s+much\s+(?:to\s+tip|(?:does|is)\s+(?:it|that)\s+cost)",
    ]
    if any(re.search(p, text) for p in transport_patterns):
        return SupervisorDecision(
            target="tourist_help",
            confidence=0.87,
            is_egypt_scoped=True,
            reasoning="User asked about transport options or tipping — factual tourist information."
        )

    # 5. Planner (New Itinerary Generation) — only explicit multi-day planning
    planner_keywords = ["itinerary", "day trip", "schedule", "budget for"]
    planner_patterns = [
        r"\b(?:plan|create|make|design|build)\s+(?:a\s+)?(?:\d+[-\s]day|my|a|the)\s+(?:trip|itinerary|tour|visit|schedule)",
        r"\b\d+\s+days?\s+in\s+\w+",
    ]
    if any(re.search(rf"\b{re.escape(pw)}\b", text) for pw in planner_keywords):
        return SupervisorDecision(
            target="planner",
            confidence=0.92,
            is_egypt_scoped=True,
            reasoning="User requested travel planning or multi-day itinerary generation for Egypt."
        )
    if any(re.search(p, text) for p in planner_patterns):
        return SupervisorDecision(
            target="planner",
            confidence=0.90,
            is_egypt_scoped=True,
            reasoning="User requested a structured multi-day trip plan."
        )

    # 6. Ambiguous / Short greeting -> Profiling
    return SupervisorDecision(
        target="profiling",
        confidence=0.50,
        is_egypt_scoped=True,
        reasoning="Ambiguous or introductory intent with insufficient details. Routing to profiling interviewer."
    )


def _message_text(message: Any) -> str:
    if isinstance(message, tuple) and len(message) >= 2:
        return str(message[1])
    if isinstance(message, dict):
        return str(message.get("content", message.get("text", "")))
    if hasattr(message, "content"):
        return str(message.content)
    return str(message)


def _message_role(message: Any) -> str:
    if isinstance(message, tuple) and len(message) >= 1:
        return str(message[0])
    if isinstance(message, dict):
        role = message.get("role")
        if role:
            return str(role)
        message_type = message.get("type")
        if message_type:
            return str(message_type)
    if hasattr(message, "type"):
        return str(message.type)
    return "message"


def _is_human_message(message: Any) -> bool:
    if isinstance(message, HumanMessage):
        return True
    return _message_role(message).strip().lower() in {"human", "user"}

SUBGRAPHS = {"planner", "tourist_help", "execution", "feedback"}

def supervisor_router_node(state: AppState) -> Dict[str, Any]:
    """
    Supervisor Router Node.
    Parses initial requests, maps them to primary subgraphs using structured output,
    and applies confidence gating.
    """
    messages = state.get("messages", [])
    recent_messages = messages[-3:] if messages else []
    user_prompt = next(
        (
            _message_text(message).strip()
            for message in reversed(messages)
            if _is_human_message(message) and _message_text(message).strip()
        ),
        "",
    )

    if not user_prompt:
        decision = SupervisorDecision(
            target="profiling",
            confidence=0.0,
            is_egypt_scoped=True,
            reasoning="No user text was provided.",
        )
        return {
            "next_agent": "profiling",
            "target_subgraph": "",
            "intent": decision.reasoning,
            "routing_reason": "Empty or whitespace-only input; routing to profiling.",
            "confidence": decision.confidence,
            "is_egypt_scoped": decision.is_egypt_scoped,
        }

    recent_context = "\n".join(
        f"{_message_role(message)}: {_message_text(message).strip()}"
        for message in recent_messages
        if _message_text(message).strip()
    )

    completed_steps = list(state.get("completed_steps", []))
    handoff_count = int(state.get("handoff_count", 0))
    max_handoff_count = int(state.get("max_handoff_count", 4))

    prev_subgraph = state.get("target_subgraph")
    if prev_subgraph in SUBGRAPHS:
        if prev_subgraph not in completed_steps:
            completed_steps.append(prev_subgraph)
        handoff_count += 1
    if handoff_count >= max_handoff_count:
        return {
            "next_agent": "FINISH",
            "target_subgraph": "",
            "completed_steps": completed_steps,
            "handoff_count": handoff_count,
            "intent": "Maximum subgraph handoff count reached.",
            "routing_reason": (
                f"Hard loop cap reached at {handoff_count}/{max_handoff_count} "
                "handoffs; stopping routing."
            ),
            "confidence": 1.0,
            "is_egypt_scoped": state.get("is_egypt_scoped", True),
        }

    incoming_next_agent = state.get("next_agent")
    approval_status = state.get("approval_status")

    # Problem 7: Approved itinerary -> trigger execution subgraph instead of re-classifying planning prompt
    if (
        incoming_next_agent == "execution_pending"
        or (prev_subgraph == "planner" and approval_status == "approved")
    ):
        if "execution" in completed_steps:
            return {
                "next_agent": "FINISH",
                "target_subgraph": "",
                "completed_steps": completed_steps,
                "handoff_count": handoff_count,
                "intent": "Execution already completed for approved itinerary.",
                "routing_reason": "Execution already in completed_steps; finishing.",
                "confidence": 1.0,
                "is_egypt_scoped": state.get("is_egypt_scoped", True),
            }
        return {
            "next_agent": "execution",
            "target_subgraph": "execution",
            "completed_steps": completed_steps,
            "handoff_count": handoff_count,
            "intent": "Approved itinerary ready for execution.",
            "routing_reason": "Itinerary approved by user; routing to execution subgraph.",
            "confidence": 1.0,
            "is_egypt_scoped": True,
        }

    # Problem 7: Rejected itinerary -> finish cleanly without re-running planner
    if incoming_next_agent == "human_approval_completed" and approval_status == "rejected":
        return {
            "next_agent": "FINISH",
            "target_subgraph": "",
            "completed_steps": completed_steps,
            "handoff_count": handoff_count,
            "intent": "Itinerary was rejected by user.",
            "routing_reason": "User rejected itinerary approval; ending turn without execution.",
            "confidence": 1.0,
            "is_egypt_scoped": state.get("is_egypt_scoped", True),
        }
    
    decision: Optional[SupervisorDecision] = None
    router_llm = get_router_llm()

    if router_llm is not None:
        try:
            structured_llm = router_llm.with_structured_output(SupervisorDecision)
            prompt_input = (
                f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
                f"Recent conversation:\n{recent_context or '(none)'}\n\n"
                f"Latest user message: {user_prompt}"
            )
            decision = structured_llm.invoke(prompt_input)
        except Exception as exc:
            logger.warning("Supervisor router LLM failed; using heuristic fallback: %s", exc)

    if decision is None:
        decision = heuristic_fallback_classifier(user_prompt)

    target = decision.target
    if not decision.is_egypt_scoped:
        target = "out_of_scope"
    elif target in SUBGRAPHS and decision.confidence < CONFIDENCE_THRESHOLD:
        target = "profiling"

    if target in completed_steps and target != "execution":
        target = "FINISH"

    target_subgraph = target if target in SUBGRAPHS else ""
    return {
        "next_agent": target,
        "target_subgraph": target_subgraph,
        "completed_steps": completed_steps,
        "handoff_count": handoff_count,
        "intent": decision.reasoning,
        "routing_reason": (
            f"{decision.reasoning} (confidence={decision.confidence:.2f}, "
            f"threshold={CONFIDENCE_THRESHOLD:.2f})"
        ),
        "confidence": decision.confidence,
        "is_egypt_scoped": decision.is_egypt_scoped,
    }


# =========================================================================
# Item 3: Conditional Fallback Edge & Out-of-Scope Responder Node
# =========================================================================




def supervisor_conditional_edge(state: AppState) -> str:
    target = state.get("next_agent", "profiling")
    if target == "FINISH":
        return "FINISH"
    if target == "out_of_scope":
        return "out_of_scope"
    if target == "profiling":
        return "profiling"
    if target in ["planner", "tourist_help", "execution", "feedback"]:
        return target
    return "profiling"


def out_of_scope_responder_node(state: AppState) -> Dict[str, Any]:
    """
    Handles out-of-scope queries by returning a courteous refusal explaining
    Ra7al's Egypt-exclusive mission.
    """
    ai_msg = AIMessage(content=OUT_OF_SCOPE_RESPONSE)
    return {
        "messages": [ai_msg],
        "response": OUT_OF_SCOPE_RESPONSE,
        "rejection_reason": state.get("intent", "Destination outside Egypt."),
        "next_agent": "END",
        "is_egypt_scoped": False
    }







