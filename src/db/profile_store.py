"""
Ra7al AI Concierge - Long-Term Memory (LTM) Profile Store
Module: src/db/profile_store.py
Story Reference: Story 5.3 - Long-Term Preference Learning (Week 6)
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from src.state.schema import (
    UserPreferences,
    DietaryPreference,
    UserInsight,
    ExtractedPreferences,
)
from src.state.memory import MOCK_USER_DB
#UPDATE 21/11: ADDED A USER DB DATA/RA7AL.DB
from src.db.repositories.user_repo import SQLiteUserRepository


# In-memory vector index for user preference insights: user_id -> List[Dict]
USER_INSIGHTS_STORE: Dict[str, List[Dict[str, Any]]] = {}


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot_product))


class ProfileStore:
    """
    Manages persistent storage, retrieval, and vector search of user profiles
    and long-term travel preferences.
    """
    user_repo = None

    @classmethod
    def get_user_profile(cls, user_id: str) -> Optional[UserPreferences]:
        """Fetches the user's persistent profile as a Pydantic model."""
        repo = cls.user_repo or SQLiteUserRepository()
        data = repo.get_profile(user_id)
        if data is None:
            repo_user = repo.get_by_user_id(user_id) or repo.get_by_username(user_id)
            if repo_user:
                data = {
                    "user_id": repo_user.user_id,
                    "name": repo_user.name,
                    "nationality": repo_user.nationality or "Egyptian",
                    "dietary": [DietaryPreference.NONE],
                    "budget_level": "moderate",
                    "interests": [],
                    "accessibility_needs": [],
                    "travel_habits": [],
                }
                data = repo.upsert_profile(data)
        if data:
            MOCK_USER_DB[user_id] = data

        if not data:
            return None
        return UserPreferences(
            user_id=data["user_id"],
            name=data.get("name", "Traveler"),
            nationality=data.get("nationality", "unknown"),
            dietary=data.get("dietary", [DietaryPreference.NONE]),
            budget_level=data.get("budget_level", "moderate"),
            interests=data.get("interests", []),
            accessibility_needs=data.get("accessibility_needs", []),
        )

    @classmethod
    def ensure_user_profile(cls, user_id: str, name: str = "Traveler") -> UserPreferences:
        """Retrieves existing profile or initializes a new default profile in store."""
        profile = cls.get_user_profile(user_id)
        if profile:
            return profile

        default_data = {
            "user_id": user_id,
            "name": name,
            "nationality": "Egyptian",
            "dietary": [DietaryPreference.NONE],
            "budget_level": "moderate",
            "interests": [],
            "accessibility_needs": [],
            "travel_habits": [],
        }
        repo = cls.user_repo or SQLiteUserRepository()
        data = repo.upsert_profile(default_data)
        MOCK_USER_DB[user_id] = data
        return UserPreferences(**data)

    @classmethod
    def update_user_preferences(
        cls,
        user_id: str,
        extracted: ExtractedPreferences
    ) -> UserPreferences:
        """
        Persistently updates the user's Long-Term Memory (LTM) profile with
        newly learned dietary rules, interests, budget tier, habits, and accessibility.
        """
        # Ensure profile exists
        cls.ensure_user_profile(user_id)
        repo = cls.user_repo or SQLiteUserRepository()
        user_entry = repo.get_profile(user_id)
        if not user_entry:
            user_entry = cls.ensure_user_profile(user_id).model_dump()

        # 1. Update Dietary Preferences
        if extracted.dietary:
            current_dietary = set(user_entry.get("dietary", []))
            # Remove NONE if specific preferences are added
            if DietaryPreference.NONE in current_dietary and extracted.dietary:
                current_dietary.discard(DietaryPreference.NONE)
            for d in extracted.dietary:
                current_dietary.add(d)
            user_entry["dietary"] = list(current_dietary)

        # 2. Update Interests
        if extracted.interests:
            current_interests = set(user_entry.get("interests", []))
            current_interests.update(extracted.interests)
            user_entry["interests"] = list(current_interests)

        # 3. Update Budget Level
        if extracted.budget_level:
            user_entry["budget_level"] = extracted.budget_level

        # 4. Update Accessibility Needs
        if extracted.accessibility_needs:
            current_access = set(user_entry.get("accessibility_needs", []))
            current_access.update(extracted.accessibility_needs)
            user_entry["accessibility_needs"] = list(current_access)

        # 5. Update Travel Habits (e.g. morning flights)
        if extracted.travel_habits:
            current_habits = set(user_entry.get("travel_habits", []))
            current_habits.update(extracted.travel_habits)
            user_entry["travel_habits"] = list(current_habits)

        # 6. Index insights into vector store
        for insight in extracted.insights:
            cls.store_insight_embedding(user_id, insight)

        repo.upsert_profile(user_entry)
        MOCK_USER_DB[user_id] = user_entry
        return cls.get_user_profile(user_id)

    @classmethod
    def store_insight_embedding(cls, user_id: str, insight: UserInsight) -> None:
        """Stores insight metadata and vector embedding in the user's vector index."""
        if user_id not in USER_INSIGHTS_STORE:
            USER_INSIGHTS_STORE[user_id] = []

        # Avoid exact duplicate entries
        existing = [
            i for i in USER_INSIGHTS_STORE[user_id]
            if i.get("key") == insight.key and i.get("insight_type") == insight.insight_type
        ]
        if existing:
            existing[0]["value"] = insight.value
            existing[0]["raw_text"] = insight.raw_text
            existing[0]["embedding"] = insight.embedding
            existing[0]["tags"] = insight.tags
        else:
            USER_INSIGHTS_STORE[user_id].append(insight.model_dump())

    @classmethod
    def search_user_insights(
        cls,
        user_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search over the user's historical preference insights.
        """
        from src.agents.feedback.preference_learning import compute_insight_embedding

        records = USER_INSIGHTS_STORE.get(user_id, [])
        if not records:
            return []

        query_vec = compute_insight_embedding(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for record in records:
            rec_vec = record.get("embedding")
            if rec_vec:
                sim = cosine_similarity(query_vec, rec_vec)
                scored.append((sim, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**rec, "similarity_score": round(score, 4)}
            for score, rec in scored[:top_k]
        ]
