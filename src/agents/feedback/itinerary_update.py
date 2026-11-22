"""
Active Itinerary Modifier (Story 5.2)
Takes trip-specific modification requests (e.g., "Change dinner to an Italian restaurant"),
updates the current plan state, and loops through the pre-existing Planner validation
loops (Story 3.2, Story 3.3, Story 3.4) to ensure the modified plan remains feasible.
Module: src/agents/feedback/itinerary_update.py
"""

import copy
import re
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.messages import AIMessage

from src.state.schema import (
    GeneratedPlan,
    ItineraryDay,
    TripRequirements,
    UserPreferences,
    PlanModification,
)
from src.agents.planner.constraint_validator import (
    validate_budget,
    validate_logistics,
)
from src.agents.planner.temporal_validator import (
    validate_temporal,
    validate_accessibility,
)
from src.agents.planner.evaluator import evaluate_plan
from src.agents.planner.repairs import repair_budget_violations


# Egyptian dining and attraction catalog for smart replacements
CUISINE_REPLACEMENTS: Dict[str, Dict[str, Any]] = {
    "italian": {
        "name": "Bella Italian Trattoria (Zamalek)",
        "category": "dining",
        "cuisine": "Italian",
        "cost_usd": 18.0,
        "duration_hours": 1.5,
        "start_time": "19:30",
        "end_time": "21:00",
        "opening_time": "12:00",
        "closing_time": "23:30",
        "travel_from_prev_mins": 15,
        "accessibility": ["wheelchair access", "elevator"],
        "description": "Authentic wood-fired Italian pizzas, pasta, and Mediterranean dishes."
    },
    "seafood": {
        "name": "Fish Market Mediterranean Seafood",
        "category": "dining",
        "cuisine": "Seafood",
        "cost_usd": 22.0,
        "duration_hours": 1.5,
        "start_time": "19:30",
        "end_time": "21:00",
        "opening_time": "12:00",
        "closing_time": "23:30",
        "travel_from_prev_mins": 20,
        "accessibility": ["wheelchair access"],
        "description": "Fresh catch of the day with stunning waterfront views."
    },
    "vegetarian": {
        "name": "Koshary Abou Tarek & Vegetarian Delights",
        "category": "dining",
        "cuisine": "Vegetarian",
        "cost_usd": 6.0,
        "duration_hours": 1.0,
        "start_time": "19:00",
        "end_time": "20:30",
        "opening_time": "10:00",
        "closing_time": "23:00",
        "travel_from_prev_mins": 15,
        "accessibility": ["wheelchair access", "step-free"],
        "description": "100% plant-based traditional Egyptian koshary and lentil specialties."
    },
    "vegan": {
        "name": "Osana Vegan Wellness Cafe (Maadi)",
        "category": "dining",
        "cuisine": "Vegan",
        "cost_usd": 12.0,
        "duration_hours": 1.5,
        "start_time": "19:00",
        "end_time": "20:30",
        "opening_time": "09:00",
        "closing_time": "22:00",
        "travel_from_prev_mins": 20,
        "accessibility": ["wheelchair access"],
        "description": "Organic, whole-food plant-based bowls, fresh juices, and vegan desserts."
    },
    "traditional": {
        "name": "Farahat Traditional Egyptian Kitchen",
        "category": "dining",
        "cuisine": "Egyptian",
        "cost_usd": 14.0,
        "duration_hours": 1.5,
        "start_time": "19:30",
        "end_time": "21:00",
        "opening_time": "12:00",
        "closing_time": "01:00",
        "travel_from_prev_mins": 15,
        "accessibility": ["wheelchair access"],
        "description": "Famous stuffed pigeon, grilled meats, and authentic Egyptian stews."
    }
}

ATTRACTION_REPLACEMENTS: Dict[str, Dict[str, Any]] = {
    "felucca": {
        "name": "Sunset Nile Felucca Sailing",
        "category": "activity",
        "cost_usd": 15.0,
        "duration_hours": 1.5,
        "start_time": "17:00",
        "end_time": "18:30",
        "opening_time": "08:00",
        "closing_time": "20:00",
        "travel_from_prev_mins": 15,
        "accessibility": ["step-free"],
        "description": "Peaceful sailing tour on the Nile in a traditional wooden sailboat."
    },
    "museum": {
        "name": "Grand Egyptian Museum (GEM)",
        "category": "cultural",
        "cost_usd": 25.0,
        "duration_hours": 3.0,
        "start_time": "09:30",
        "end_time": "12:30",
        "opening_time": "09:00",
        "closing_time": "18:00",
        "travel_from_prev_mins": 20,
        "accessibility": ["wheelchair access", "elevator", "step-free"],
        "description": "World-class museum featuring King Tutankhamun's complete treasures."
    }
}


def parse_modification_request(prompt: str, current_plan: Optional[GeneratedPlan] = None) -> PlanModification:
    """
    Parses conversational modification requests into structured PlanModification parameters.
    """
    text = prompt.lower().strip()

    # 1. Day Extraction
    # TODO (Problem 15): The modifier invents things: defaults to Day 1 when no day is specified.
    # If the user says "Change dinner to an Italian restaurant", it arbitrarily modifies Day 1 dinner
    # instead of checking active context or asking the user which day they wish to modify.
    day_match = re.search(r"\bday\s*(\d+)\b", text)
    day_num = int(day_match.group(1)) if day_match else 1
    if current_plan and current_plan.days:
        if day_num < 1 or day_num > len(current_plan.days):
            day_num = 1

    # 2. Action Detection
    action = "replace"
    if any(kw in text for kw in ["remove", "delete", "cancel", "drop", "skip"]):
        action = "remove"
    elif any(kw in text for kw in ["add", "include", "insert"]):
        action = "add"
    elif any(kw in text for kw in ["reschedule", "earlier", "later", "move to"]):
        action = "reschedule"

    # 3. Time Slot / Target Activity Detection
    time_slot = None
    target = None
    if "dinner" in text or "evening" in text:
        time_slot = "evening"
        target = "dinner"
    elif "lunch" in text or "afternoon" in text:
        time_slot = "afternoon"
        target = "lunch"
    elif "breakfast" in text or "morning" in text:
        time_slot = "morning"
        target = "breakfast"

    # Extract target entity from verbs like remove, change, replace
    target_match_kw = re.search(
        r"\b(?:remove|delete|cancel|drop|skip|replace|change|swap)\s+(?:the\s+)?([a-z0-9\s'-]+?)(?:\s+on\s+day|\s+with|\s+to|\s+for|$)",
        text
    )
    if target_match_kw:
        extracted_target = target_match_kw.group(1).strip()
        # Filter out common stop words
        if extracted_target not in ["dinner", "lunch", "breakfast", "activity", "itinerary", "plan"]:
            target = extracted_target

    # 4. Replacement Details
    replacement_name = None
    replacement_category = None
    cost_usd = None

    for cuisine, item in CUISINE_REPLACEMENTS.items():
        if cuisine in text:
            replacement_name = item["name"]
            replacement_category = item["category"]
            cost_usd = item["cost_usd"]
            if not time_slot:
                time_slot = "evening"
            break

    if not replacement_name:
        for attr_key, item in ATTRACTION_REPLACEMENTS.items():
            if attr_key in text:
                replacement_name = item["name"]
                replacement_category = item["category"]
                cost_usd = item["cost_usd"]
                break

    if not replacement_name and action == "replace":
        # Check if there is a 'to/with <replacement>' clause
        to_with_match = re.search(r"\b(?:to|with)\s+(?:an?\s+)?([a-z0-9\s'-]+)", text)
        if to_with_match:
            clean_name = to_with_match.group(1).strip()
            replacement_name = clean_name.title()
        else:
            # TODO (Problem 15): The modifier invents things: creates arbitrary "Custom: ..." places
            # with hardcoded $15.0 cost and hardcodes accessibility: ["wheelchair access", "step-free"].
            # Should look up verified Egypt places or validate accessibility attributes properly.
            clean_prompt = re.sub(r"^(please\s+|can we\s+|i want to\s+|change\s+|modify\s+)", "", text).strip()
            replacement_name = f"Custom: {clean_prompt.title()}"
        replacement_category = "dining" if time_slot == "evening" else "activity"
        cost_usd = 15.0

    return PlanModification(
        action=action,
        day_number=day_num,
        time_slot=time_slot,
        target_place_or_activity=target,
        replacement_name=replacement_name,
        replacement_category=replacement_category,
        estimated_cost_usd=cost_usd,
        notes=prompt
    )


def apply_modification_to_plan(
    plan: GeneratedPlan,
    modification: PlanModification
) -> Tuple[GeneratedPlan, str]:
    """
    Applies the PlanModification to the GeneratedPlan in memory.
    Returns: (updated_plan, change_summary)
    """
    updated_days: List[ItineraryDay] = []
    change_summary = ""

    target_day = modification.day_number or 1

    for day in plan.days:
        if day.day_number != target_day:
            updated_days.append(day.model_copy(deep=True))
            continue

        new_activities: List[Dict[str, Any]] = []
        applied = False

        if modification.action == "replace":
            # 1. Determine index to replace
            replace_idx = None
            target_str = (modification.target_place_or_activity or "").lower()

            for i, act in enumerate(day.activities):
                act_name = act.get("name", "").lower()
                act_cat = act.get("category", "").lower()
                act_slot = act.get("time_slot", "").lower()

                # Exact or substring match on target or slot
                if target_str and (target_str in act_name or target_str in act_cat):
                    replace_idx = i
                    break
                if modification.time_slot and act_slot == modification.time_slot.lower():
                    replace_idx = i
                    break

            # TODO (Problem 15): The modifier invents things: when target activity doesn't match anything,
            # it defaults to replacing the last activity (replace_idx = len(day.activities) - 1).
            # If target cannot be found, it should ask the user for clarification rather than silently overwriting.
            if replace_idx is None and day.activities:
                replace_idx = len(day.activities) - 1

            for i, act in enumerate(day.activities):
                if i == replace_idx and not applied:
                    rep_name = modification.replacement_name or "Custom Experience"
                    rep_cat = modification.replacement_category or act.get("category", "activity")
                    rep_cost = modification.estimated_cost_usd or act.get("cost_usd", 15.0)

                    template = None
                    for item in {**CUISINE_REPLACEMENTS, **ATTRACTION_REPLACEMENTS}.values():
                        if item["name"].lower() == rep_name.lower():
                            template = item
                            break

                    new_act = dict(act)
                    new_act["name"] = rep_name
                    new_act["category"] = rep_cat
                    new_act["cost_usd"] = rep_cost
                    if template:
                        new_act["description"] = template.get("description", new_act.get("description", ""))
                        if "accessibility" in template:
                            new_act["accessibility"] = template["accessibility"]
                        if "start_time" in template and "end_time" in template:
                            new_act["start_time"] = template["start_time"]
                            new_act["end_time"] = template["end_time"]
                        if "opening_time" in template and "closing_time" in template:
                            new_act["opening_time"] = template["opening_time"]
                            new_act["closing_time"] = template["closing_time"]

                    new_activities.append(new_act)
                    applied = True
                    change_summary = (
                        f"Day {day.day_number}: Replaced '{act.get('name')}' with '{rep_name}' "
                        f"(${rep_cost:.2f} USD)."
                    )
                else:
                    new_activities.append(dict(act))

        elif modification.action == "remove":
            target_str = (modification.target_place_or_activity or "").lower()
            remove_idx = None

            for i, act in enumerate(day.activities):
                act_name = act.get("name", "").lower()
                act_cat = act.get("category", "").lower()
                act_slot = act.get("time_slot", "").lower()

                if target_str and (target_str in act_name or target_str in act_cat):
                    remove_idx = i
                    break
                if modification.time_slot and act_slot == modification.time_slot.lower():
                    remove_idx = i
                    break

            for i, act in enumerate(day.activities):
                if i == remove_idx and not applied:
                    applied = True
                    change_summary = f"Day {day.day_number}: Removed '{act.get('name')}' from schedule."
                    continue
                new_activities.append(dict(act))

        elif modification.action == "add":
            new_activities = [dict(a) for a in day.activities]
            rep_name = modification.replacement_name or "Additional Activity"
            rep_cost = modification.estimated_cost_usd or 15.0
            new_act = {
                "time_slot": modification.time_slot or "afternoon",
                "name": rep_name,
                "category": modification.replacement_category or "activity",
                "cost_usd": rep_cost,
                "duration_hours": 1.5,
                "start_time": "16:00",
                "end_time": "17:30",
                "opening_time": "09:00",
                "closing_time": "21:00",
                "travel_from_prev_mins": 15,
                "accessibility": ["wheelchair access", "step-free"],
                "description": f"Custom added activity: {rep_name}."
            }
            new_activities.append(new_act)
            change_summary = f"Day {day.day_number}: Added '{rep_name}' (${rep_cost:.2f} USD)."

        else:
            new_activities = [dict(a) for a in day.activities]
            change_summary = f"Day {day.day_number}: Updated schedule details."

        # Recalculate day estimated cost
        day_cost = sum(
            act.get("cost_usd", 0.0) + act.get("transit_cost_usd", 0.0)
            for act in new_activities
        )
        updated_days.append(ItineraryDay(
            day_number=day.day_number,
            activities=new_activities,
            estimated_cost=round(day_cost, 2)
        ))

    modified_plan = GeneratedPlan(
        plan_id=plan.plan_id or "plan_modified",
        title=plan.title or "Updated Egypt Itinerary",
        days=updated_days,
        status="modified"
    )

    return modified_plan, change_summary


def validate_plan_through_planner_loops(
    plan: GeneratedPlan,
    requirements: Optional[TripRequirements],
    preferences: Optional[UserPreferences]
) -> Dict[str, Any]:
    """
    Executes the pre-existing Planner validation loops (Stories 3.2, 3.3, 3.4):
    1. Budget & Logistics Validation
    2. Temporal & Accessibility Validation
    3. Plan Quality Evaluation & Grading
    4. Budget Repair loop if overage occurs
    """
    max_budget = getattr(requirements, "total_budget", None) or 99999.0

    # 1. Story 3.2: Budget & Logistics
    budget_valid, budget_err, total_cost, overage = validate_budget(plan, max_budget)
    logistics_valid, logistics_errs = validate_logistics(plan)

    # TODO (Problem 15): The modifier claims to run planner validation loops, but it only ever calls
    # repair_budget_violations. It does not attempt to repair temporal collisions, opening-hour mismatches,
    # or accessibility violations. Implement full multi-constraint repair loops.
    # If budget violation and repairable, attempt repair
    if not budget_valid and overage > 0:
        plan = repair_budget_violations(plan, max_budget, overage)
        budget_valid, budget_err, total_cost, overage = validate_budget(plan, max_budget)

    constraint_errors: List[str] = []
    if not budget_valid and budget_err:
        constraint_errors.append(budget_err)
    if not logistics_valid:
        constraint_errors.extend(logistics_errs)
    is_constraint_valid = len(constraint_errors) == 0

    # 2. Story 3.3: Temporal & Accessibility
    temporal_valid, temporal_errs = validate_temporal(plan)
    accessibility_valid, accessibility_errs = validate_accessibility(plan, preferences)

    advanced_errors = temporal_errs + accessibility_errs
    is_advanced_valid = len(advanced_errors) == 0

    # 3. Story 3.4: Plan Quality Evaluation
    all_errors = constraint_errors + advanced_errors
    evaluation = evaluate_plan(plan, all_errors)
    plan_quality = evaluation.get("grade", "Good")
    is_executable = (plan_quality == "Good")

    plan.status = "validated" if is_executable else "review"

    return {
        "validated_plan": plan,
        "is_constraint_valid": is_constraint_valid,
        "constraint_errors": constraint_errors,
        "is_advanced_valid": is_advanced_valid,
        "temporal_errors": temporal_errs,
        "accessibility_errors": accessibility_errs,
        "advanced_errors": advanced_errors,
        "total_cost": total_cost,
        "budget_overage": overage,
        "evaluation": evaluation,
        "plan_quality": plan_quality,
        "is_executable": is_executable,
    }


def itinerary_modifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Active Itinerary Modifier (Story 5.2):
    Parses modification from chat, updates the current_plan, executes
    Planner validation loops, and outputs confirmation to user.
    """
    plan = state.get("current_plan")
    if not plan or not isinstance(plan, GeneratedPlan) or not plan.days:
        msg = (
            "⚠️ There is currently no active itinerary in your session to modify. "
            "Please ask me to create an itinerary first (e.g. 'Plan 3 days in Cairo')!"
        )
        ai_msg = AIMessage(content=msg)
        return {
            **state,
            "response": msg,
            "messages": state.get("messages", []) + [ai_msg],
            "next_agent": "feedback_completed"
        }

    # Extract user prompt
    # TODO (Problem 11): The "both" feedback path is broken.
    # Here messages[-1] is read. In the compound "both" flow, ltm_preference_update_node just executed and appended
    # an AIMessage ("✅ Saved your preference..."). Consequently, messages[-1] is that AI confirmation,
    # NOT the user's modification request! Filter messages for the last HumanMessage or user message tuple.
    messages = state.get("messages", [])
    user_prompt = ""
    if messages:
        last = messages[-1]
        user_prompt = last[1] if isinstance(last, tuple) else getattr(last, "content", str(last))
    elif state.get("user_query"):
        user_prompt = str(state.get("user_query"))

    # 1. Parse modification request
    mod = parse_modification_request(user_prompt, plan)

    # 2. Apply modification to plan
    updated_plan, change_summary = apply_modification_to_plan(plan, mod)

    # 3. Validate modified plan through Week 3/4 Planner validation loops
    reqs = state.get("trip_requirements")
    prefs = state.get("user_preferences")
    val_results = validate_plan_through_planner_loops(updated_plan, reqs, prefs)

    validated_plan = val_results["validated_plan"]
    is_valid = val_results["is_executable"]

    # 4. Formulate response message
    if is_valid:
        confirmation = (
            f"✅ **Itinerary Updated Successfully!**\n\n"
            f"{change_summary}\n\n"
            f"• **New Total Estimated Cost:** ${val_results['total_cost']:.2f} USD\n"
            f"• **Planner Validation:** Passed all budget, temporal, and logistics checks.\n"
            f"• **Status:** Ready for your review."
        )
    else:
        confirmation = (
            f"⚠️ **Itinerary Modified with Warnings**\n\n"
            f"{change_summary}\n\n"
            f"• **Total Cost:** ${val_results['total_cost']:.2f} USD\n"
            f"• **Validation Note:** {'; '.join(val_results['advanced_errors'] + val_results['constraint_errors'])}\n"
            f"• **Status:** Review required."
        )

    ai_msg = AIMessage(content=confirmation)

    return {
        **state,
        "current_plan": validated_plan,
        "is_constraint_valid": val_results["is_constraint_valid"],
        "constraint_errors": val_results["constraint_errors"],
        "is_advanced_valid": val_results["is_advanced_valid"],
        "temporal_errors": val_results["temporal_errors"],
        "accessibility_errors": val_results["accessibility_errors"],
        "total_cost": val_results["total_cost"],
        "evaluation": val_results["evaluation"],
        "plan_quality": val_results["plan_quality"],
        "is_executable": is_valid,
        "plan_modification": mod.model_dump(),
        "response": confirmation,
        "messages": state.get("messages", []) + [ai_msg],
        "next_agent": "feedback_completed"
    }
