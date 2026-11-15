"""
Profiling & Interview Agent (Story 5.1 / Dev 1)
Identifies missing constraints, generates targeted clarifying questions,
and updates session preferences in short-term state.
Module: src/agents/feedback/profiling.py
"""

from typing import Dict, Any, List, Optional
import re
from langchain_core.messages import AIMessage
from src.state.schema import TripRequirements, UserPreferences


PROFILING_GREETING_PROMPT = (
    "Ahlan! Welcome to Ra7al (رحال), your personal AI travel concierge for Egypt.\n\n"
)


def extract_constraints_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts travel planning constraints (destination, duration, budget) from natural language.
    """
    extracted: Dict[str, Any] = {}
    lower_text = text.lower().strip()

    # 1. Destination Matching
    destinations = [
        "cairo", "giza", "luxor", "aswan", "alexandria",
        "hurghada", "sharm el sheikh", "dahab", "siwa"
    ]
    for dest in destinations:
        if dest in lower_text:
            extracted["destination"] = dest.title()
            break

    # 2. Duration Matching (e.g. "3 days", "5-day trip", "1 week")
    day_match = re.search(r"\b(\d+)\s*(?:-| )*(?:day|days|night|nights)\b", lower_text)
    if day_match:
        extracted["duration_days"] = int(day_match.group(1))
    elif "week" in lower_text:
        extracted["duration_days"] = 7

    # 3. Budget Matching (e.g. "$500", "500 usd", "budget of 300")
    budget_match = re.search(r"(?:\$|usd\s*|budget\s*(?:of|is)?\s*\$?)\s*(\d+(?:,\d+)?(?:\.\d+)?)", lower_text)
    if budget_match:
        extracted["total_budget"] = float(budget_match.group(1).replace(",", ""))
    elif "budget" in lower_text:
        extracted["budget_level"] = "budget"
        extracted["total_budget"] = 250.0
    elif "luxury" in lower_text:
        extracted["budget_level"] = "luxury"
        extracted["total_budget"] = 1500.0

    return extracted


def detect_missing_constraints(requirements: Optional[TripRequirements]) -> List[str]:
    """
    Evaluates TripRequirements to identify critical missing constraints.
    Returns a list of missing field names: e.g. ['destination', 'duration', 'budget'].
    """
    if not requirements:
        return ["destination", "duration", "budget"]

    missing: List[str] = []
    
    # Destination check
    if not requirements.destination or requirements.destination.strip().lower() in ["", "egypt"]:
        missing.append("destination")

    # Budget check
    if requirements.total_budget is None or requirements.total_budget <= 0:
        missing.append("budget")

    # Duration check (either explicit start/end dates or duration specified)
    has_dates = bool(requirements.start_date and requirements.end_date)
    has_duration = hasattr(requirements, "duration_days") and getattr(requirements, "duration_days", None)
    if not (has_dates or has_duration):
        missing.append("duration")

    return missing


def ask_user_questions(missing_fields: List[str]) -> str:
    """
    Generates targeted, courteous clarifying questions based on missing constraints.
    """
    if not missing_fields:
        return "Thank you! I have all the details needed to design your itinerary."

    prompts = {
        "destination": "• Which city or region in Egypt would you like to visit (e.g., Cairo, Luxor, Aswan, Alexandria, Red Sea)?",
        "duration": "• How many days do you plan to spend on this trip?",
        "budget": "• What is your total approximate budget for this trip (in USD), or your preferred travel tier (budget, moderate, luxury)?"
    }

    questions = [prompts[field] for field in missing_fields if field in prompts]
    return (
        "To help me craft the best possible itinerary for you, could you please provide a few more details?\n\n"
        + "\n".join(questions)
    )


def update_session_preferences(state: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hydrates or updates `trip_requirements` in state with provided user inputs.
    """
    current_reqs = state.get("trip_requirements")
    if isinstance(current_reqs, TripRequirements):
        req_data = current_reqs.model_dump()
    elif isinstance(current_reqs, dict):
        req_data = dict(current_reqs)
    else:
        req_data = {}

    req_data.update(user_answers)
    
    # Ensure destination has default if none
    destination = req_data.get("destination", "Cairo")
    
    updated_req = TripRequirements(
        destination=destination,
        duration_days=req_data.get("duration_days"),
        start_date=req_data.get("start_date"),
        end_date=req_data.get("end_date"),
        total_budget=req_data.get("total_budget"),
        companions_count=req_data.get("companions_count", 1),
        current_location=req_data.get("current_location")
    )

    return {
        **state,
        "trip_requirements": updated_req
    }


def profiling_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Checks if active planning constraints are fulfilled.
    If constraints are missing, extracts what it can from the last user message.
    If still missing critical parameters, sets `needs_profiling = True` and prompts the user.
    """
    # 1. Attempt extracting from user prompt if requirements aren't fully set
    messages = state.get("messages", [])
    last_text = ""
    if messages:
        last_msg = messages[-1]
        last_text = last_msg[1] if isinstance(last_msg, tuple) else getattr(last_msg, "content", str(last_msg))

    current_reqs = state.get("trip_requirements")
    extracted = extract_constraints_from_text(last_text) if last_text else {}

    # Merge extracted constraints into requirements
    if extracted:
        state = update_session_preferences(state, extracted)
        current_reqs = state.get("trip_requirements")

    missing = detect_missing_constraints(current_reqs)

    if missing:
        clarifying_question = ask_user_questions(missing)
        ai_msg = AIMessage(content=clarifying_question)
        return {
            **state,
            "messages": state.get("messages", []) + [ai_msg],
            "response": clarifying_question,
            "missing_constraints": missing,
            "needs_profiling": True,
            "next_agent": "profiling"
        }

    return {
        **state,
        "needs_profiling": False,
        "missing_constraints": [],
        "next_agent": "draft_generator"
    }


def profiling_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interactive interview agent node.
    Greets user and triggers clarifying questions when information is incomplete.
    """
    res = profiling_check_node(state)
    clarifying = res.get("response", "")
    full_response = f"{PROFILING_GREETING_PROMPT}{clarifying}"
    ai_msg = AIMessage(content=full_response)

    return {
        **res,
        "messages": state.get("messages", []) + [ai_msg],
        "response": full_response,
        "next_agent": "profiling_completed"
    }
