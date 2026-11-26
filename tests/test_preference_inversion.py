import pytest

from src.db.repositories.user_repo import SQLiteUserRepository, UserRecord
from src.services.profile_service import (
    ProfileService,
    merge_extracted,
    merge_extracted_with_changes,
)
from src.state.schema import (
    DietaryPreference,
    ExtractedPreferences,
    PreferenceItem,
    UserPreferences,
)


def test_dislike_museums_moves_to_avoided():
    profile = UserPreferences(user_id="u", name="U", liked_interests=["museums"])
    extracted = ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums", polarity="avoid")
    ])
    result = merge_extracted(profile, extracted)
    assert result.liked_interests == []
    assert result.avoided_interests == ["museums"]


def test_changes_include_legacy_additions_and_item_inversion():
    profile = UserPreferences(user_id="u", name="U")
    extracted = ExtractedPreferences(
        interests=["museums"],
        items=[PreferenceItem(category="interest", value="museums", polarity="avoid")],
    )

    result = merge_extracted_with_changes(profile, extracted)

    assert result.profile.liked_interests == []
    assert result.profile.avoided_interests == ["museums"]
    assert result.changes == ["liked: museums", "avoided: museums (was liked)"]


def test_interest_polarity_inverts_and_plural_singular_matches():
    profile = UserPreferences(user_id="u", name="U")
    items = [
        PreferenceItem(category="interest", value="museums", polarity="like"),
        PreferenceItem(category="interest", value="museum", polarity="avoid"),
        PreferenceItem(category="interest", value="museums", polarity="like"),
    ]

    result = merge_extracted(profile, ExtractedPreferences(items=items))

    assert result.liked_interests == ["museums"]
    assert result.avoided_interests == []


def test_dietary_inversion_and_unknown_values():
    profile = UserPreferences(user_id="u", name="U", dietary=[DietaryPreference.VEGAN])
    avoided = merge_extracted(profile, ExtractedPreferences(items=[
        PreferenceItem(category="dietary", value="vegan", polarity="avoid")
    ]))
    liked = merge_extracted(avoided, ExtractedPreferences(items=[
        PreferenceItem(category="dietary", value="halal", polarity="like")
    ]))
    unknown = merge_extracted(liked, ExtractedPreferences(items=[
        PreferenceItem(category="dietary", value="pescatarian", polarity="like")
    ]))

    assert avoided.dietary == [DietaryPreference.NONE]
    assert liked.dietary == [DietaryPreference.HALAL]
    assert unknown.dietary == [DietaryPreference.HALAL]


def test_accessibility_habit_and_budget_polarity(caplog):
    profile = UserPreferences(
        user_id="u",
        name="U",
        accessibility_needs=["wheelchair access"],
        travel_habits=["morning flights"],
    )
    extracted = ExtractedPreferences(items=[
        PreferenceItem(category="accessibility", value="wheelchair access", polarity="avoid"),
        PreferenceItem(category="travel_habit", value="morning flights", polarity="avoid"),
        PreferenceItem(category="budget_level", value="luxury", polarity="like"),
        PreferenceItem(category="budget_level", value="budget", polarity="avoid"),
    ])

    result = merge_extracted(profile, extracted)

    assert result.accessibility_needs == []
    assert result.travel_habits == []
    assert result.budget_level == "luxury"
    assert "Removing accessibility need" in caplog.text


def test_this_trip_items_are_skipped_and_apply_is_idempotent():
    profile = UserPreferences(user_id="u", name="U")
    transient = PreferenceItem(
        category="interest", value="museums", polarity="avoid", scope="this_trip"
    )
    assert merge_extracted(profile, ExtractedPreferences(items=[transient])) == profile

    long_term = ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums", polarity="like")
    ])
    once = merge_extracted(profile, long_term)
    assert merge_extracted(once, long_term) == once


def test_registered_inversion_persists_across_service_instances(tmp_path):
    db_path = str(tmp_path / "profiles.db")
    repo = SQLiteUserRepository(db_path=db_path)
    repo.create(UserRecord(
        user_id="u", username="u", name="Traveler", hashed_password="pw"
    ))
    first = ProfileService(repo)
    first.get_or_init("u")
    first.apply_extracted("u", ExtractedPreferences(interests=["museums"]))
    first.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museum", polarity="avoid")
    ]))

    restored = ProfileService(SQLiteUserRepository(db_path=db_path)).get("u")

    assert restored.liked_interests == []
    assert restored.avoided_interests == ["museums"]


def test_guest_inversion_and_forget_both_interest_lists(tmp_path):
    service = ProfileService(SQLiteUserRepository(db_path=str(tmp_path / "guest.db")))
    profile = UserPreferences(
        user_id="guest", name="Guest", liked_interests=["museums"],
        avoided_interests=["crowds"],
    )
    result = service.apply_extracted(
        None,
        ExtractedPreferences(items=[
            PreferenceItem(category="interest", value="museum", polarity="avoid")
        ]),
        current=profile,
    )
    assert result.liked_interests == []
    assert result.avoided_interests == ["crowds", "museums"]

    service.repo.create(UserRecord(
        user_id="u", username="u", name="User", hashed_password="pw"
    ))
    service.get_or_init("u")
    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums", polarity="avoid")
    ]))
    forgotten = service.forget("u", "interest", "museum")
    assert forgotten.liked_interests == []
    assert forgotten.avoided_interests == []