"""
Supervisor Request Routing Engine (Story 1.2 / Dev 4)
Item 1: Supervisor Prompt Engineering & Intent Definition
"""

from typing import Literal
from pydantic import BaseModel, Field

# The target subgraphs and outcomes recognized by the Supervisor
TargetSubgraph = Literal[
    "planner",       # Multi-day itineraries, custom tour plans, activity scheduling
    "tourist_help",  # Visas, emergency numbers, local transport rules, nearby recommendations
    "execution",     # Booking flights/buses/trains, calendar syncing
    "feedback",      # Modifying existing confirmed plans, rating past activities
    "profiling",     # Onboarding, gathering preferences, handling vague/ambiguous intents
    "out_of_scope"   # Requests outside Egypt or non-travel queries
]


class SupervisorDecision(BaseModel):
    """Structured output for the Supervisor Router Agent decision."""
    target: TargetSubgraph = Field(
        ...,
        description=(
            "The designated target subgraph for the user query: "
            "'planner' for planning/itineraries, "
            "'tourist_help' for immediate help/visas/emergency/local spots, "
            "'execution' for booking/calendar, "
            "'feedback' for itinerary adjustments/feedback, "
            "'profiling' for ambiguous/vague input needing interview, "
            "'out_of_scope' for non-Egypt travel or prohibited queries."
        )
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the intent classification from 0.0 to 1.0."
    )
    is_egypt_scoped: bool = Field(
        ...,
        description=(
            "True if the query pertains to travel within or to Egypt (or is a general greeting). "
            "False if the user requests travel to other countries (e.g. Paris, Tokyo, Dubai)."
        )
    )
    reasoning: str = Field(
        ...,
        description="Concise rationale explaining the classification and scope verification."
    )


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
   - Multi-day itineraries (e.g., "Plan a 3-day trip to Cairo", "5 days in Luxor and Aswan").
   - Itinerary design based on budget, pace, family, or specific dates in Egypt.
   - Day-by-day scheduling of sights, temples, and activities.

2. "tourist_help":
   - Immediate on-the-ground traveler assistance, local rules, emergency numbers (Tourist Police, Ambulance).
   - Egyptian eVisa and Visa on Arrival eligibility requirements.
   - Local cultural rules, tipping (baksheesh), scams, SIM cards, or dress codes.
   - Spontaneous hyper-local recommendations (e.g., "Where can I eat Koshary near Tahrir Square right now?", "Best cafes in Zamalek").

3. "execution":
   - Explicit intent to book or reserve flights, buses (GoBus), trains (ENR), or hotels.
   - Calendar synchronization requests (e.g., "Sync this schedule to my Google Calendar").

4. "feedback":
   - Modifying or tweaking an existing generated itinerary (e.g., "Remove the museum on day 2", "I prefer vegetarian food instead").
   - Submitting reviews, ratings, or feedback about past trip activities.

5. "profiling":
   - Ambiguous, vague, or generic queries that lack clear travel details (e.g., "Hi", "Help me plan something", "I want to travel", "What can you do?").
   - When the user's intent is unclear or confidence is below 0.65, route to profiling so the interviewer agent can gather preferences.

6. "out_of_scope":
   - Non-Egypt travel requests or non-travel queries (e.g., coding, math, general chat).

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
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))


def get_router_llm():
    """
    Instantiate the chat model with structured output capability.
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


def heuristic_fallback_classifier(user_text: str) -> SupervisorDecision:
    """
    Deterministic backup classifier used if LLM API is unavailable or in offline testing.
    Guarantees robust routing under zero network connectivity.
    """
    text = user_text.lower().strip()
    
    # 1. Geographic Exclusion (Non-Egypt destinations)
    non_egypt_keywords = ["paris", "dubai", "tokyo", "london", "rome", "new york", "bali", "turkey", "spain", "france"]
    for kw in non_egypt_keywords:
        if kw in text:
            return SupervisorDecision(
                target="out_of_scope",
                confidence=1.0,
                is_egypt_scoped=False,
                reasoning=f"User requested '{kw}', which is outside Egypt."
            )

    # 2. Execution / Booking / Calendar
    booking_keywords = ["book", "reserve", "ticket", "calendar", "sync", "flight ticket", "gobus"]
    if any(bw in text for bw in booking_keywords):
        return SupervisorDecision(
            target="execution",
            confidence=0.90,
            is_egypt_scoped=True,
            reasoning="User explicitly requested booking or calendar synchronization."
        )

    # 3. Feedback / Modifying plan (Checked before Planner so modifications aren't mistaken for new plans)
    feedback_keywords = ["change", "modify", "remove", "don't like", "update", "replace", "rating", "review"]
    if any(fw in text for fw in feedback_keywords):
        return SupervisorDecision(
            target="feedback",
            confidence=0.85,
            is_egypt_scoped=True,
            reasoning="User requested changes to an active travel plan or provided feedback."
        )

    # 4. Tourist Help / Visas / Emergencies / Nearby
    help_keywords = ["visa", "emergency", "police", "embassy", "dress code", "scam", "tipping", "koshary", "near me", "nearby", "recommend", "sim card"]
    if any(hw in text for hw in help_keywords):
        return SupervisorDecision(
            target="tourist_help",
            confidence=0.88,
            is_egypt_scoped=True,
            reasoning="User requested immediate tourist advice, visa information, emergency help, or nearby POIs."
        )

    # 5. Planner (New Itinerary Generation)
    planner_keywords = ["itinerary", "plan", "days in", "day trip", "visit cairo", "visit luxor", "aswan", "alexandria", "schedule", "budget for"]
    if any(pw in text for pw in planner_keywords):
        return SupervisorDecision(
            target="planner",
            confidence=0.92,
            is_egypt_scoped=True,
            reasoning="User requested travel planning or multi-day itinerary generation for Egypt."
        )

    # 6. Ambiguous / Short greeting -> Profiling
    return SupervisorDecision(
        target="profiling",
        confidence=0.50,
        is_egypt_scoped=True,
        reasoning="Ambiguous or introductory intent with insufficient details. Routing to profiling interviewer."
    )


def supervisor_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor Router Node.
    Parses initial requests, maps them to primary subgraphs using structured output,
    and applies confidence gating.
    """
    messages = state.get("messages", [])
    if not messages:
        user_prompt = "Hello"
    else:
        last_msg = messages[-1]
        # Support tuple ('user', text), string, or LangChain BaseMessage
        if isinstance(last_msg, tuple):
            user_prompt = last_msg[1]
        elif hasattr(last_msg, "content"):
            user_prompt = last_msg.content
        else:
            user_prompt = str(last_msg)

    decision: Optional[SupervisorDecision] = None
    llm = get_router_llm()

    if llm:
        try:
            structured_llm = llm.with_structured_output(SupervisorDecision)
            system_msg = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
            user_msg = HumanMessage(content=f"User Request: {user_prompt}")
            decision = structured_llm.invoke([system_msg, user_msg])
        except Exception as e:
            print(f"[Supervisor] LLM invocation failed, using fallback: {e}")
            decision = heuristic_fallback_classifier(user_prompt)
    else:
        decision = heuristic_fallback_classifier(user_prompt)

    # Enforce confidence threshold & fallback
    selected_target: TargetSubgraph = decision.target
    if decision.target != "out_of_scope" and decision.confidence < CONFIDENCE_THRESHOLD:
        selected_target = "profiling"
        decision.reasoning += f" (Low confidence {decision.confidence:.2f} < threshold {CONFIDENCE_THRESHOLD}. Fallback to profiling.)"

    # Enforce geographic exclusion
    if not decision.is_egypt_scoped:
        selected_target = "out_of_scope"

    return {
        **state,
        "next_agent": selected_target,
        "intent": decision.reasoning,
        "confidence": decision.confidence,
        "is_egypt_scoped": decision.is_egypt_scoped
    }


# =========================================================================
# Item 3: Conditional Fallback Edge & Out-of-Scope Responder Node
# =========================================================================

from langchain_core.messages import AIMessage


def supervisor_conditional_edge(state: Dict[str, Any]) -> str:
     
    target = state.get("next_agent", "profiling")
    is_egypt = state.get("is_egypt_scoped", True)
    confidence = state.get("confidence", 1.0)

    # 1. Geographic Boundary Gate
    if not is_egypt or target == "out_of_scope":
        return "out_of_scope"

    # 2. Confidence Fallback Gate (routes ambiguous/low-confidence queries to Profiling Agent)
    if confidence < CONFIDENCE_THRESHOLD or target == "profiling":
        return "profiling"

    # 3. Valid Primary Subgraph Routes
    if target in ["planner", "tourist_help", "execution", "feedback"]:
        return target

    # Default safe fallback
    return "profiling"


def out_of_scope_responder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles out-of-scope queries by returning a courteous refusal explaining
    Ra7al's Egypt-exclusive mission.
    """
    ai_msg = AIMessage(content=OUT_OF_SCOPE_RESPONSE)
    return {
        **state,
        "messages": state.get("messages", []) + [ai_msg],
        "response": OUT_OF_SCOPE_RESPONSE,
        "rejection_reason": state.get("intent", "Destination outside Egypt."),
        "next_agent": "END",
        "is_egypt_scoped": False
    }



def create_supervisor_graph():
    """
    Build and compile the Supervisor LangGraph workflow with conditional routing
    and fallback edges.
    """
    from langgraph.graph import StateGraph, START, END
    from src.agents.feedback.profiling import profiling_agent_node

    workflow = StateGraph(dict)

    # 1. Add nodes
    workflow.add_node("supervisor", supervisor_router_node)
    workflow.add_node("out_of_scope", out_of_scope_responder_node)
    workflow.add_node("profiling", profiling_agent_node)

    # 2. Add edges & conditional routing
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_conditional_edge,
        {
            "out_of_scope": "out_of_scope",
            "profiling": "profiling",
            "planner": END,
            "tourist_help": END,
            "execution": END,
            "feedback": END,
        }
    )
    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("profiling", END)

    return workflow.compile()



