"""
Ra7al AI Concierge - Long-Term Memory (LTM) Profile Store
Module: src/db/profile_store.py
Story Reference: Story 5.3 - Long-Term Preference Learning (Week 6)
"""

from typing import Any, Dict, List, Optional
from src.state.schema import (
    UserPreferences,
    UserInsight,
    ExtractedPreferences,
)
from src.services.profile_service import ProfileService, UnknownUserError

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot_product))


class ProfileStore:
    """
    DEPRECATED: Thin compatibility wrapper over ProfileService.
    Manages persistent storage, retrieval, and token-overlap search of user profiles
    and long-term travel preferences.
    """
    user_repo = None

    @classmethod
    def _get_service(cls) -> ProfileService:
        if cls.user_repo is not None:
            return ProfileService(cls.user_repo)
        return ProfileService.default()

    @classmethod
    def get_user_profile(cls, user_id: str) -> Optional[UserPreferences]:
        """DEPRECATED: Fetches the user's persistent profile as a Pydantic model."""
        svc = cls._get_service()
        return svc.get_or_init(user_id)

    @classmethod
    def ensure_user_profile(cls, user_id: str, name: str = "Traveler") -> UserPreferences:
        """
        DEPRECATED: Retrieves existing profile or initializes a new default profile in store.
        Raises UnknownUserError if the user is not registered.
        """
        svc = cls._get_service()
        if not svc.is_registered(user_id):
            raise UnknownUserError(f"User '{user_id}' is not registered.")
        profile = svc.get_or_init(user_id)
        if profile is None:
            raise UnknownUserError(f"User '{user_id}' is not registered.")
        return profile

    @classmethod
    def update_user_preferences(
        cls,
        user_id: str,
        extracted: ExtractedPreferences
    ) -> UserPreferences:
        """
        DEPRECATED: Persistently updates the user's Long-Term Memory (LTM) profile with
        newly learned dietary rules, interests, budget tier, habits, and accessibility.
        """
        svc = cls._get_service()
        return svc.apply_extracted(user_id, extracted)

    @classmethod
    def store_insight_embedding(cls, user_id: str, insight: UserInsight) -> None:
        """DEPRECATED: Persists insight metadata without storing hashed embeddings."""
        cls._get_service().apply_extracted(
            user_id,
            ExtractedPreferences(insights=[insight]),
        )

    @classmethod
    def search_user_insights(
        cls,
        user_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Searches persisted insights using token overlap.
        """
        return cls._get_service().search_insights(user_id, query, top_k)
