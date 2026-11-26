import logging
from typing import Any, Dict, Optional
from src.db.repositories.user_repo import SQLiteUserRepository
from src.state.schema import DietaryPreference, ExtractedPreferences, UserPreferences

logger = logging.getLogger(__name__)
GUEST_ID = "guest"


class UnknownUserError(Exception):
    pass


def _norm(user_id: Optional[str]) -> str:
    return (user_id or "").strip().lower()


def merge_extracted(p: UserPreferences, ex: ExtractedPreferences) -> UserPreferences:
    """PURE function, no DB. Used for both registered users and guests."""
    dietary = list(dict.fromkeys([*p.dietary, *ex.dietary]))
    if ex.dietary:
        dietary = [d for d in dietary if d != DietaryPreference.NONE]
    return p.model_copy(update={
        "dietary": dietary,
        "liked_interests": sorted(set(p.liked_interests) | set(ex.interests)),
        "budget_level": ex.budget_level or p.budget_level,
        "accessibility_needs": sorted(set(p.accessibility_needs or []) | set(ex.accessibility_needs)),
        "travel_habits": sorted(set(p.travel_habits) | set(ex.travel_habits)),
    })
    # TODO(Day 3, #4): apply ex.items with polarity (like/avoid inversion, removal)


class ProfileService:
    _default: Optional["ProfileService"] = None

    def __init__(self, repo: Optional[SQLiteUserRepository] = None):
        self.repo = repo or SQLiteUserRepository()   # ONE repo instance per service

    @classmethod
    def default(cls) -> "ProfileService":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def set_default(cls, svc: Optional["ProfileService"]) -> None:   # tests
        cls._default = svc

    # identity
    def is_registered(self, user_id: Optional[str]) -> bool:
        """True only if a row exists in `users`."""
        uid = _norm(user_id)
        if not uid:
            return False
        if self.repo.get_by_user_id(uid):
            return True
        user_by_name = self.repo.get_by_username(uid)
        if user_by_name:
            logger.warning("username used as user_id (deprecated): %s", uid)
            return True
        return False

    @staticmethod
    def guest_profile() -> UserPreferences:
        """fresh object each call, never shared."""
        return UserPreferences(user_id=GUEST_ID, name="Guest")

    # reads
    def get(self, user_id: Optional[str]) -> Optional[UserPreferences]:
        """NO side effects, never creates rows."""
        uid = _norm(user_id)
        if not uid:
            return None
        data = self.repo.get_profile(uid)
        if data:
            return UserPreferences.model_validate(data)
        user_by_name = self.repo.get_by_username(uid)
        if user_by_name:
            logger.warning("username used as user_id (deprecated): %s", uid)
            data = self.repo.get_profile(user_by_name.user_id)
            if data:
                return UserPreferences.model_validate(data)
        return None

    def get_or_init(self, user_id: Optional[str]) -> Optional[UserPreferences]:
        """Existing profile; or a default profile created for a REGISTERED user; else None."""
        uid = _norm(user_id)
        if not uid:
            return None
        user = self.repo.get_by_user_id(uid)
        if not user:
            user = self.repo.get_by_username(uid)
            if user:
                logger.warning("username used as user_id (deprecated): %s", uid)
        if not user:
            return None

        data = self.repo.get_profile(user.user_id)
        if data:
            return UserPreferences.model_validate(data)

        default_pref = UserPreferences(
            user_id=user.user_id,
            name=user.name,
            nationality=user.nationality,
            dietary=[DietaryPreference.NONE],
            budget_level="moderate",
            liked_interests=[],
            avoided_interests=[],
            accessibility_needs=[],
            travel_habits=[],
        )
        self.repo.upsert_profile(default_pref.model_dump(mode="json"))
        return default_pref

    # writes
    def apply_extracted(
        self,
        user_id: Optional[str],
        extracted: ExtractedPreferences,
        current: Optional[UserPreferences] = None
    ) -> UserPreferences:
        """Registered -> persisted via repo.update_profile_atomic (merge_extracted inside).
        Guest/unknown/empty id -> merge_extracted on `current or guest_profile()`; NO DB access."""
        if extracted.items:
            logger.debug("PreferenceItem entries in extracted.items ignored for now (Day 3): %s", extracted.items)

        uid = _norm(user_id)
        if not uid:
            base = current or self.guest_profile()
            return merge_extracted(base, extracted)

        user = self.repo.get_by_user_id(uid)
        if not user:
            user = self.repo.get_by_username(uid)
            if user:
                logger.warning("username used as user_id (deprecated): %s", uid)

        if not user:
            base = current or self.guest_profile()
            return merge_extracted(base, extracted)

        existing = self.repo.get_profile(user.user_id)
        if not existing:
            self.get_or_init(user.user_id)

        def mutate(row_dict: Dict[str, Any]) -> Dict[str, Any]:
            pref = UserPreferences.model_validate(row_dict)
            merged = merge_extracted(pref, extracted)
            return merged.model_dump(mode="json")

        updated_dict = self.repo.update_profile_atomic(user.user_id, mutate)
        if updated_dict:
            return UserPreferences.model_validate(updated_dict)

        base = self.get_or_init(user.user_id) or self.guest_profile()
        return merge_extracted(base, extracted)
