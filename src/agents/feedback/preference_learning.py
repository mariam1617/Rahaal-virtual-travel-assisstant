"""
Insight Extractor Pipeline (Story 5.3)
Distills conversational preference feedback into structured tags,
dietary requirements, travel habits, and vector embeddings for long-term learning.
Module: src/agents/feedback/preference_learning.py
"""

import hashlib
import math
import os
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage

from src.state.schema import (
    DietaryPreference,
    UserInsight,
    ExtractedPreferences,
)


from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_gemini_embedder():
    """Initializes and caches GoogleGenerativeAIEmbeddings if GOOGLE_API_KEY is available."""
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        logger.warning(f"Could not initialize Gemini embeddings, falling back to hash vector: {e}")
        return None


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2 normalizes a vector to unit length for accurate cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [round(x / norm, 6) for x in vec]
    return [0.0] * len(vec)


def _fallback_hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Fallback deterministic vector when offline or GOOGLE_API_KEY is not set."""
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower().strip())
    for i, token in enumerate(tokens):
        token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = token_hash % dim
        weight = 1.0 / (1.0 + 0.1 * i)
        vector[idx] += weight
        if i > 0:
            bigram = f"{tokens[i-1]}_{token}"
            bigram_hash = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            vector[bigram_hash % dim] += 1.5 * weight

    return _l2_normalize(vector)


def compute_insight_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Computes a deterministic, unit-normalized semantic embedding vector using
    Gemini text-embedding-004 (sliced & L2-normalized to `dim`), with a deterministic
    offline fallback for test environments without API keys.
    """
    clean_text = text.lower().strip()
    if not clean_text:
        return [0.0] * dim

    embedder = _get_gemini_embedder()
    if embedder is not None:
        try:
            raw_vec = embedder.embed_query(clean_text)
            # Matryoshka truncation/padding to match EMBEDDING_DIM (384) and re-normalize
            resized = (raw_vec + [0.0] * dim)[:dim]
            return _l2_normalize(resized)
        except Exception as e:
            logger.warning(f"Embedding API call failed, using fallback: {e}")

    return _fallback_hash_embedding(clean_text, dim)


def extract_preference_insights(text: str) -> ExtractedPreferences:
    """
    Distills user chat text into structured tags, habits, dietary choices,
    budget tiers, and vector embeddings.
    """
    lower = text.lower().strip()
    insights: List[UserInsight] = []
    dietary_list: List[DietaryPreference] = []
    interests_list: List[str] = []
    budget_level: Optional[str] = None
    accessibility_list: List[str] = []
    habits_list: List[str] = []

    # 1. Dietary Preference Extraction
    if re.search(r"\bvegan\b", lower):
        dietary_list.append(DietaryPreference.VEGAN)
        insights.append(UserInsight(
            insight_type="dietary",
            key="vegan",
            value="vegan",
            tags=["dietary", "vegan", "plant-based"],
            raw_text="Prefers 100% vegan dining.",
            confidence=0.95,
            embedding=compute_insight_embedding("vegan plant-based dining")
        ))
    elif re.search(r"\bvegetarian\b", lower):
        dietary_list.append(DietaryPreference.VEGETARIAN)
        insights.append(UserInsight(
            insight_type="dietary",
            key="vegetarian",
            value="vegetarian",
            tags=["dietary", "vegetarian", "meatless"],
            raw_text="Prefers vegetarian food options.",
            confidence=0.95,
            embedding=compute_insight_embedding("vegetarian meatless dining")
        ))

    if re.search(r"\bhalal\b", lower):
        dietary_list.append(DietaryPreference.HALAL)
        insights.append(UserInsight(
            insight_type="dietary",
            key="halal",
            value="halal",
            tags=["dietary", "halal", "certified"],
            raw_text="Requires halal food certification.",
            confidence=0.95,
            embedding=compute_insight_embedding("halal certified food")
        ))

    # 2. Travel Habits & Flight Preferences
    if re.search(r"\bmorning\s*flights?\b", lower):
        habits_list.append("morning flights")
        insights.append(UserInsight(
            insight_type="travel_habit",
            key="morning_flights",
            value="morning flights",
            tags=["flights", "timing", "morning", "travel_habit"],
            raw_text="Strongly prefers morning flight schedules.",
            confidence=0.92,
            embedding=compute_insight_embedding("morning flight schedule departure early")
        ))
    elif re.search(r"\bevening\s*flights?\b", lower):
        habits_list.append("evening flights")
        insights.append(UserInsight(
            insight_type="travel_habit",
            key="evening_flights",
            value="evening flights",
            tags=["flights", "timing", "evening", "travel_habit"],
            raw_text="Prefers evening or night flights.",
            confidence=0.90,
            embedding=compute_insight_embedding("evening flight departure night")
        ))

    if re.search(r"\bdirect\s*flights?\b", lower):
        habits_list.append("direct flights")
        insights.append(UserInsight(
            insight_type="travel_habit",
            key="direct_flights",
            value="direct flights",
            tags=["flights", "nonstop", "direct"],
            raw_text="Prefers direct nonstop flights.",
            confidence=0.90,
            embedding=compute_insight_embedding("direct nonstop flights")
        ))

    if re.search(r"\bwindow\s*seat\b", lower):
        habits_list.append("window seat")
        insights.append(UserInsight(
            insight_type="travel_habit",
            key="window_seat",
            value="window seat",
            tags=["flights", "seating", "window"],
            raw_text="Prefers window seating.",
            confidence=0.90,
            embedding=compute_insight_embedding("window seat airplane")
        ))

    # 3. Accommodation & Budget Tiers
    if re.search(r"\b(luxury|5-star|five\s*star|high-end)\b", lower):
        budget_level = "luxury"
        habits_list.append("luxury hotels")
        insights.append(UserInsight(
            insight_type="budget_level",
            key="luxury",
            value="luxury",
            tags=["budget", "luxury", "5-star", "upscale"],
            raw_text="Prefers luxury accommodations and top-tier services.",
            confidence=0.90,
            embedding=compute_insight_embedding("luxury 5-star hotel upscale travel")
        ))
    elif re.search(r"\b(budget|backpacker|hostel|cheap)\b", lower):
        budget_level = "budget"
        habits_list.append("budget accommodations")
        insights.append(UserInsight(
            insight_type="budget_level",
            key="budget",
            value="budget",
            tags=["budget", "affordable", "backpacking"],
            raw_text="Prefers affordable, cost-effective options.",
            confidence=0.88,
            embedding=compute_insight_embedding("budget travel backpacker affordable")
        ))

    # 4. Interests & Activity Passions
    # TODO (Problem 12): Preferences can be saved backwards.
    # Input like "I don't like museums, remove it" stores "museums" as an active interest because there
    # is no negation detection and no like/avoid polarity in the schema.
    # Implement negation parsing (e.g. "don't like", "avoid", "hate", "no") and split interests into
    # `liked_interests` vs `avoided_interests` (or add an `avoid` polarity flag).
    interest_keywords = {
        "historical sites": ["historical", "ancient", "temple", "pharaoh", "pyramid", "ruins"],
        "museums": ["museum", "antiquities", "exhibits", "artifacts"],
        "local food": ["street food", "local food", "culinary", "cooking", "koshary"],
        "snorkeling": ["snorkeling", "scuba", "diving", "coral reef", "red sea"],
        "coastal cities": ["beach", "sea", "resort", "hurghada", "sharm", "dahab", "alexandria"],
        "nightlife": ["nightlife", "clubs", "bars", "evening entertainment"],
        "photography": ["photography", "scenic views", "sunset", "photo spots"],
    }

    for interest_name, kw_list in interest_keywords.items():
        if any(kw in lower for kw in kw_list):
            interests_list.append(interest_name)
            insights.append(UserInsight(
                insight_type="interest",
                key=interest_name,
                value=interest_name,
                tags=["interest", interest_name.replace(" ", "_")],
                raw_text=f"Interested in {interest_name}.",
                confidence=0.88,
                embedding=compute_insight_embedding(f"travel interest {interest_name}")
            ))

    # 5. Accessibility Needs
    if re.search(r"\b(wheelchair|wheelchair\s*access|mobility\s*impairment)\b", lower):
        accessibility_list.append("wheelchair access")
        insights.append(UserInsight(
            insight_type="accessibility",
            key="wheelchair_access",
            value="wheelchair access",
            tags=["accessibility", "wheelchair", "step-free"],
            raw_text="Requires wheelchair accessible venues and transportation.",
            confidence=0.95,
            embedding=compute_insight_embedding("wheelchair accessible handicap ramp elevator")
        ))

    # Fallback generic insight if no specific patterns matched but text has substance
    if not insights and len(lower.split()) >= 2:
        clean_note = re.sub(r"^(i prefer|i like|keep in mind that)\s*", "", lower).strip()
        generic_key = re.sub(r"[^\w\s]", "", clean_note).replace(" ", "_")[:30]
        insights.append(UserInsight(
            insight_type="general_preference",
            key=generic_key,
            value=clean_note,
            tags=["general_preference", "user_feedback"],
            raw_text=clean_note,
            confidence=0.75,
            embedding=compute_insight_embedding(clean_note)
        ))

    return ExtractedPreferences(
        dietary=dietary_list,
        interests=interests_list,
        budget_level=budget_level,
        accessibility_needs=accessibility_list,
        travel_habits=habits_list,
        insights=insights
    )


def insight_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Insight Extractor Pipeline (Story 5.3):
    Distills user input text into structured tags, habits, and vector embeddings,
    saving them to pending state for the LTM Profile Update node.
    """
    messages = state.get("messages", [])
    user_prompt = ""
    if messages:
        last = messages[-1]
        user_prompt = last[1] if isinstance(last, tuple) else getattr(last, "content", str(last))
    elif state.get("user_query"):
        user_prompt = str(state.get("user_query"))

    # Extract structured preferences
    extracted = extract_preference_insights(user_prompt)

    # Convert insights to serializable dicts
    insights_dicts = [i.model_dump() for i in extracted.insights]

    return {
        "extracted_insights": insights_dicts,
        "pending_preferences": extracted.model_dump(),
        "next_agent": "ltm_updater",
    }


def ltm_preference_update_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Long-Term Preference Update (Story 5.3):
    Saves extracted insights persistently into the user's Long-Term Memory (LTM)
    profile store and updates active session state.
    """
    from src.db.profile_store import ProfileStore

    # TODO (Problem 13): All users share one profile because anonymous or unauthenticated users
    # default to user_id = "test_user". Use session-scoped ephemeral profiles for guests or enforce auth.
    user_id = state.get("user_id") or "test_user"

    # Hydrate ExtractedPreferences object
    pending = state.get("pending_preferences")
    if pending and isinstance(pending, dict):
        extracted = ExtractedPreferences(**pending)
    else:
        messages = state.get("messages", [])
        prompt = ""
        if messages:
            last = messages[-1]
            prompt = last[1] if isinstance(last, tuple) else getattr(last, "content", str(last))
        extracted = extract_preference_insights(prompt)

    # Persistently update LTM profile store
    updated_profile = ProfileStore.update_user_preferences(user_id, extracted)

    # Highlight learned features for the user
    learned_items: List[str] = []
    if extracted.travel_habits:
        learned_items.extend(extracted.travel_habits)
    if extracted.dietary:
        learned_items.extend([d.value for d in extracted.dietary])
    if extracted.budget_level:
        learned_items.append(f"{extracted.budget_level} tier")
    if extracted.interests:
        learned_items.extend(extracted.interests)
    if extracted.accessibility_needs:
        learned_items.extend(extracted.accessibility_needs)

    summary_phrase = ", ".join(learned_items) if learned_items else "preferences"

    fb_type = state.get("feedback_type", "long_term_preference")

    if fb_type == "both":
        # Compound intent: preference saved, proceed to modify the itinerary
        next_node = "itinerary_modifier"
        confirm_msg = (
            f"✅ Saved your preference for **{summary_phrase}** to your travel profile. "
            f"Now updating your active itinerary..."
        )
    else:
        next_node = "feedback_completed"
        confirm_msg = (
            f"✅ **Profile Updated!** I've recorded your preference for **{summary_phrase}** "
            f"in your travel profile. Future itineraries and recommendations will automatically reflect this!"
        )

    ai_msg = AIMessage(content=confirm_msg)

    return {
        "user_preferences": updated_profile,
        "response": confirm_msg,
        "messages": [ai_msg],
        "next_agent": next_node,
    }
