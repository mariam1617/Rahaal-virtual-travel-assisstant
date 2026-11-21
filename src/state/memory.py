from typing import Optional, Dict
from src.state.schema import UserPreferences, DietaryPreference


#UPDATE 21/11: ADDED A USER DB DATA/RA7AL.DB
# # Mock Database: بيانات تجريبية لمستخدم وتفضيلاته
# MOCK_USER_DB: Dict[str, dict] = {
#     "test_user": {
#         "user_id": "test_user",
#         "name": "Test User",
#         "nationality": "Egyptian",
#         "dietary": [DietaryPreference.NONE],
#         "budget_level": "moderate",
#         "interests": [],
#         "accessibility_needs": []
#     },
#     "user_101": {
#         "user_id": "user_101",
#         "name": "سما ضياء",
#         "nationality": "Egyptian",
#         "dietary": [DietaryPreference.HALAL],
#         "budget_level": "moderate",
#         "interests": ["historical sites", "local food", "museums"],
#         "accessibility_needs": []
#     },
#     "user_202": {
#         "user_id": "user_202",
#         "name": "Alex Smith",
#         "nationality": "German",
#         "dietary": [DietaryPreference.VEGAN],
#         "budget_level": "luxury",
#         "interests": ["snorkeling", "coastal cities", "nightlife"],
#         "accessibility_needs": ["wheelchair access"]
#     }
# }
# Compatibility cache only. It is no longer the source of truth.
MOCK_USER_DB: Dict[str, dict] = {}

class MemoryLoader:
    """مسؤول عن استرجاع الذاكرة الدائمة وتفضيلات المستخدم"""
    
    @staticmethod
    def get_user_profile(user_id: str) -> Optional[UserPreferences]:
        """استرجاع الملف الدائم للمستخدم وتحويله إلى Pydantic Model"""
        repo = SQLiteUserRepository()
        data = repo.get_profile(user_id)
        if data is None:
            user = repo.get_by_user_id(user_id) or repo.get_by_username(user_id)
            if not user:
                return None
            data = {
                "user_id": user.user_id,
                "name": user.name,
                "nationality": user.nationality or "Egyptian",
                "dietary": [DietaryPreference.NONE],
                "budget_level": "moderate",
                "interests": [],
                "accessibility_needs": [],
                "travel_habits": [],
            }
            data = repo.upsert_profile(data)

        MOCK_USER_DB[data["user_id"]] = data
        return UserPreferences(**data)

    @staticmethod
    def save_user_preference(user_id: str, new_interests: list) -> bool:
        """تحديث تفضيلات المستخدم (للـ Long-Term Learning لاحقاً)"""
        profile = MemoryLoader.get_user_profile(user_id)
        if profile is None:
            return False

        data = profile.model_dump()
        data["interests"] = list(set(profile.interests).union(new_interests))
        data = SQLiteUserRepository().upsert_profile(data)
        MOCK_USER_DB[user_id] = data
        return True