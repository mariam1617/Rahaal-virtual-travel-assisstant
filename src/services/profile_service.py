import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.db.repositories.user_repo import SQLiteUserRepository
from src.state.schema import DietaryPreference, ExtractedPreferences, PreferenceItem, UserPreferences

logger = logging.getLogger(__name__)
GUEST_ID = "guest"


class UnknownUserError(Exception):
    pass


def _norm(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _key(value: str) -> str:
    """Comparison key with naive singularization for simple plurals."""
    normalized = _norm(value)
    return normalized[:-1] if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss") else normalized


def _add(values: list, value: str) -> bool:
    normalized = _norm(value)
    if _key(normalized) in {_key(existing) for existing in values}:
        return False
    values.append(normalized)
    return True


def _drop(values: list, value: str) -> bool:
    key = _key(value)
    original_length = len(values)
    values[:] = [existing for existing in values if _key(existing) != key]
    return len(values) != original_length


def _matching_value(values: list, value: str) -> Optional[str]:
    key = _key(value)
    return next((existing for existing in values if _key(existing) == key), None)


@dataclass
class ApplyResult:
    profile: UserPreferences
    changes: list[str] = field(default_factory=list)


def apply_items(profile: UserPreferences, items: list[PreferenceItem]) -> ApplyResult:
    """Apply long-term preference items without performing I/O."""
    result_profile = profile.model_copy(deep=True)
    changes: list[str] = []

    for item in items:
        value = _norm(item.value)
        if item.scope != "long_term":
            logger.debug("Skipping non-long-term preference item: %s", item.scope)
            continue
        if not value:
            logger.debug("Skipping preference item with an empty value")
            continue

        if item.category == "interest":
            if item.polarity == "like":
                previous = _matching_value(result_profile.avoided_interests, value)
                was_avoided = _drop(result_profile.avoided_interests, value)
                added = _add(result_profile.liked_interests, previous or value)
                if was_avoided or added:
                    suffix = " (was avoided)" if was_avoided else ""
                    changes.append(f"liked: {value}{suffix}")
            else:
                previous = _matching_value(result_profile.liked_interests, value)
                was_liked = _drop(result_profile.liked_interests, value)
                added = _add(result_profile.avoided_interests, previous or value)
                if was_liked or added:
                    suffix = " (was liked)" if was_liked else ""
                    changes.append(f"avoided: {value}{suffix}")
        elif item.category == "dietary":
            dietary_value = value
            try:
                preference = DietaryPreference(dietary_value)
            except ValueError:
                logger.debug("Ignoring unsupported dietary preference: %s", value)
                continue
            if item.polarity == "like":
                if preference == DietaryPreference.NONE:
                    changed = result_profile.dietary != [DietaryPreference.NONE]
                    result_profile.dietary = [DietaryPreference.NONE]
                else:
                    without_none = [d for d in result_profile.dietary if d != DietaryPreference.NONE]
                    changed = preference not in without_none
                    result_profile.dietary = list(dict.fromkeys([*without_none, preference]))
                if changed:
                    changes.append(f"diet: {dietary_value}")
            else:
                remaining = [d for d in result_profile.dietary if d != preference]
                if not remaining:
                    remaining = [DietaryPreference.NONE]
                if remaining != result_profile.dietary:
                    result_profile.dietary = remaining
                    changes.append(f"diet removed: {dietary_value}")
        elif item.category == "accessibility":
            needs = result_profile.accessibility_needs or []
            if item.polarity == "like":
                if _add(needs, value):
                    result_profile.accessibility_needs = needs
                    changes.append(f"accessibility: {value}")
            elif _drop(needs, value):
                result_profile.accessibility_needs = needs
                logger.warning("Removing accessibility need: %s", value)
                changes.append(f"accessibility removed: {value}")
        elif item.category == "travel_habit":
            if item.polarity == "like":
                if _add(result_profile.travel_habits, value):
                    changes.append(f"habit: {value}")
            elif _drop(result_profile.travel_habits, value):
                changes.append(f"habit removed: {value}")
        elif item.category == "budget_level":
            if item.polarity == "avoid":
                logger.debug("Ignoring avoid polarity for budget level: %s", value)
                continue
            if value in {"budget", "moderate", "luxury"} and result_profile.budget_level != value:
                result_profile.budget_level = value
                changes.append(f"budget: {value}")
            elif value not in {"budget", "moderate", "luxury"}:
                logger.debug("Ignoring unsupported budget level: %s", value)

    return ApplyResult(profile=result_profile, changes=changes)


def _merge_legacy(p: UserPreferences, ex: ExtractedPreferences) -> ApplyResult:
    """Apply the legacy fields and describe only effective additions."""
    before = p.model_copy(deep=True)
    dietary = list(dict.fromkeys([*p.dietary, *ex.dietary]))
    if ex.dietary:
        dietary = [d for d in dietary if d != DietaryPreference.NONE]
    merged = p.model_copy(deep=True, update={
        "dietary": dietary,
        "liked_interests": sorted(set(p.liked_interests) | set(ex.interests)),
        "budget_level": ex.budget_level or p.budget_level,
        "accessibility_needs": sorted(set(p.accessibility_needs or []) | set(ex.accessibility_needs)),
        "travel_habits": sorted(set(p.travel_habits) | set(ex.travel_habits)),
    })
    changes: list[str] = []
    for label, old_values, new_values in (
        ("diet", before.dietary, merged.dietary),
        ("liked", before.liked_interests, merged.liked_interests),
        ("accessibility", before.accessibility_needs or [], merged.accessibility_needs or []),
        ("habit", before.travel_habits, merged.travel_habits),
    ):
        old_keys = {_key(value.value if isinstance(value, DietaryPreference) else value) for value in old_values}
        for value in new_values:
            display = value.value if isinstance(value, DietaryPreference) else value
            if _key(display) not in old_keys:
                changes.append(f"{label}: {display}")
    if before.budget_level != merged.budget_level:
        changes.append(f"budget: {merged.budget_level}")
    return ApplyResult(profile=merged, changes=changes)


def merge_extracted_with_changes(p: UserPreferences, ex: ExtractedPreferences) -> ApplyResult:
    legacy = _merge_legacy(p, ex)
    applied = apply_items(legacy.profile, ex.items)
    return ApplyResult(profile=applied.profile, changes=[*legacy.changes, *applied.changes])


def merge_extracted(p: UserPreferences, ex: ExtractedPreferences) -> UserPreferences:
    """PURE function, no DB. Used for both registered users and guests."""
    return merge_extracted_with_changes(p, ex).profile


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
    def apply_extracted_with_changes(
        self,
        user_id: Optional[str],
        extracted: ExtractedPreferences,
        current: Optional[UserPreferences] = None
    ) -> ApplyResult:
        """Persist registered users' changes; keep guest changes in session state only."""
        uid = _norm(user_id)
        if not uid:
            base = current or self.guest_profile()
            return merge_extracted_with_changes(base, extracted)

        user = self.repo.get_by_user_id(uid)
        if not user:
            user = self.repo.get_by_username(uid)
            if user:
                logger.warning("username used as user_id (deprecated): %s", uid)

        if not user:
            base = current or self.guest_profile()
            return merge_extracted_with_changes(base, extracted)

        existing = self.repo.get_profile(user.user_id)
        if not existing:
            self.get_or_init(user.user_id)

        captured: list[ApplyResult] = []

        def mutate(row_dict: Dict[str, Any]) -> Dict[str, Any]:
            pref = UserPreferences.model_validate(row_dict)
            result = merge_extracted_with_changes(pref, extracted)
            captured.append(result)
            return result.profile.model_dump(mode="json")

        updated_dict = self.repo.update_profile_atomic(user.user_id, mutate)
        if updated_dict:
            result = captured[-1] if captured else ApplyResult(UserPreferences.model_validate(updated_dict))
            result.profile = UserPreferences.model_validate(updated_dict)
            return result

        base = self.get_or_init(user.user_id) or self.guest_profile()
        logger.warning("Atomic profile update returned no row for registered user %s", user.user_id)
        return merge_extracted_with_changes(base, extracted)

    def apply_extracted(
        self,
        user_id: Optional[str],
        extracted: ExtractedPreferences,
        current: Optional[UserPreferences] = None
    ) -> UserPreferences:
        return self.apply_extracted_with_changes(user_id, extracted, current).profile

    def forget(self, user_id: str, category: str, value: str) -> Optional[UserPreferences]:
        uid = _norm(user_id)
        if not uid or not self.is_registered(uid):
            return None

        user = self.repo.get_by_user_id(uid) or self.repo.get_by_username(uid)
        if not user:
            return None
        if not self.repo.get_profile(user.user_id):
            self.get_or_init(user.user_id)

        def mutate(row_dict: Dict[str, Any]) -> Dict[str, Any]:
            profile = UserPreferences.model_validate(row_dict).model_copy(deep=True)
            if category == "interest":
                _drop(profile.liked_interests, value)
                _drop(profile.avoided_interests, value)
            elif category == "dietary":
                try:
                    preference = DietaryPreference(_norm(value))
                except ValueError:
                    return profile.model_dump(mode="json")
                profile.dietary = [d for d in profile.dietary if d != preference]
                if not profile.dietary:
                    profile.dietary = [DietaryPreference.NONE]
            elif category == "accessibility":
                needs = profile.accessibility_needs or []
                _drop(needs, value)
                profile.accessibility_needs = needs
            elif category == "travel_habit":
                _drop(profile.travel_habits, value)
            elif category == "budget_level" and _key(profile.budget_level) == _key(value):
                profile.budget_level = "moderate"
            return profile.model_dump(mode="json")

        updated = self.repo.update_profile_atomic(user.user_id, mutate)
        return UserPreferences.model_validate(updated) if updated else None
