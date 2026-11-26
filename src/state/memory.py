"""
Persistent profile memory loader.
Module: src/state/memory.py
"""

from typing import Optional, List
from src.state.schema import UserPreferences, ExtractedPreferences
from src.services.profile_service import ProfileService


class MemoryLoader:
    """DEPRECATED: Thin compatibility wrapper over ProfileService."""

    @staticmethod
    def get_user_profile(user_id: str) -> Optional[UserPreferences]:
        """Load user profile as a Pydantic model via ProfileService."""
        return ProfileService.default().get_or_init(user_id)

    @staticmethod
    def save_user_preference(user_id: str, new_interests: list) -> bool:
        """Update user preferences for long-term learning."""
        svc = ProfileService.default()
        if not svc.is_registered(user_id):
            return False
        svc.apply_extracted(user_id, ExtractedPreferences(interests=new_interests))
        return True
