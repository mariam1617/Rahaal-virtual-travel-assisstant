from typing import Optional, Dict
from src.state.schema import UserPreferences, DietaryPreference

# Mock Database: بيانات تجريبية لمستخدم وتفضيلاته
MOCK_USER_DB: Dict[str, dict] = {
    "user_101": {
        "user_id": "user_101",
        "name": "سما ضياء",
        "nationality": "Egyptian",
        "dietary": [DietaryPreference.HALAL],
        "budget_level": "moderate",
        "interests": ["historical sites", "local food", "museums"],
        "accessibility_needs": []
    },
    "user_202": {
        "user_id": "user_202",
        "name": "Alex Smith",
        "nationality": "German",
        "dietary": [DietaryPreference.VEGAN],
        "budget_level": "luxury",
        "interests": ["snorkeling", "coastal cities", "nightlife"],
        "accessibility_needs": ["wheelchair access"]
    }
}

class MemoryLoader:
    """مسؤول عن استرجاع الذاكرة الدائمة وتفضيلات المستخدم"""
    
    @staticmethod
    def get_user_profile(user_id: str) -> Optional[UserPreferences]:
        """استرجاع الملف الدائم للمستخدم وتحويله إلى Pydantic Model"""
        data = MOCK_USER_DB.get(user_id)
        if not data:
            return None
        return UserPreferences(**data)

    @staticmethod
    def save_user_preference(user_id: str, new_interests: list) -> bool:
        """تحديث تفضيلات المستخدم (للـ Long-Term Learning لاحقاً)"""
        if user_id in MOCK_USER_DB:
            current_interests = set(MOCK_USER_DB[user_id]["interests"])
            current_interests.update(new_interests)
            MOCK_USER_DB[user_id]["interests"] = list(current_interests)
            return True
        return False