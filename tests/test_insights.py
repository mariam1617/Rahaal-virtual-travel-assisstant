import sqlite3
import time

import pytest

from src.db.repositories.insight_repo import SQLiteInsightRepository
from src.db.repositories.user_repo import SQLiteUserRepository, UserRecord
from src.services.profile_service import ProfileService
from src.state.schema import ExtractedPreferences, PreferenceItem, UserInsight


@pytest.fixture
def registered_service(tmp_path):
    db_path = str(tmp_path / "insights.db")
    repo = SQLiteUserRepository(db_path=db_path)
    repo.create(UserRecord(
        user_id="u", username="u", name="Traveler", hashed_password="pw"
    ))
    service = ProfileService(repo)
    service.get_or_init("u")
    return service, db_path


def test_upsert_is_idempotent_and_inversion_preserves_created_at(tmp_path):
    repo = SQLiteInsightRepository(str(tmp_path / "insights.db"))
    like = {
        "insight_type": "interest",
        "key": "museum",
        "value": "museums",
        "polarity": "like",
        "tags": ["interest", "museum"],
        "raw_text": "Likes museums",
        "confidence": 1.0,
    }

    original = repo.upsert(" U ", like)
    assert repo.upsert("u", like)["id"] == original["id"]
    assert len(repo.list("u")) == 1
    time.sleep(0.002)

    inverted = repo.upsert("u", {**like, "polarity": "avoid", "raw_text": "Avoids museums"})

    assert inverted["id"] == original["id"]
    assert inverted["polarity"] == "avoid"
    assert inverted["created_at"] == original["created_at"]
    assert inverted["updated_at"] != original["updated_at"]
    assert "embedding" not in inverted


def test_insights_persist_across_service_instances(registered_service):
    service, db_path = registered_service
    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(
            category="interest", value="museums", evidence="I enjoy museums"
        )
    ]))

    restored = ProfileService(SQLiteUserRepository(db_path=db_path))

    assert restored.get_insights("u")[0]["value"] == "museums"
    assert restored.get_insights("u")[0]["polarity"] == "like"


def test_removal_items_delete_insights_and_forget_removes_profile_value(
    registered_service,
):
    service, _ = registered_service
    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="dietary", value="vegan"),
        PreferenceItem(category="interest", value="museums"),
    ]))
    assert len(service.get_insights("u")) == 2

    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="dietary", value="vegan", polarity="avoid")
    ]))
    assert not [item for item in service.get_insights("u") if item["insight_type"] == "dietary"]

    profile = service.forget("u", "interest", "museum")
    assert "museums" not in profile.liked_interests
    assert not [item for item in service.get_insights("u") if item["insight_type"] == "interest"]


def test_preference_item_polarity_overrides_legacy_insight(registered_service):
    service, _ = registered_service
    service.apply_extracted("u", ExtractedPreferences(
        items=[PreferenceItem(category="interest", value="museums", polarity="avoid")],
        insights=[UserInsight(
            insight_type="interest", key="museums", value="museums", polarity="like"
        )],
    ))

    assert service.get_insights("u")[0]["polarity"] == "avoid"


def test_search_uses_singularized_token_overlap(registered_service):
    service, _ = registered_service
    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums", evidence="Loves museums"),
        PreferenceItem(category="travel_habit", value="morning flights"),
    ]))

    results = service.search_insights("u", "museum")

    assert results[0]["key"] == "museum"
    assert results[0]["similarity_score"] == 1.0
    assert service.search_insights("u", "volcano") == []


def test_deprecated_profile_store_wrappers_use_persisted_search(registered_service):
    from src.db.profile_store import ProfileStore

    service, _ = registered_service
    ProfileStore.user_repo = service.repo
    try:
        ProfileStore.store_insight_embedding("u", UserInsight(
            insight_type="interest", key="museums", value="museums"
        ))
        results = ProfileStore.search_user_insights("u", "museum")
    finally:
        ProfileStore.user_repo = None

    assert results[0]["key"] == "museum"


def test_guests_and_unknown_users_do_not_persist_insights(registered_service):
    service, _ = registered_service
    extracted = ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums")
    ])
    service.apply_extracted(None, extracted)
    service.apply_extracted("unknown-user", extracted)

    assert service.get_insights(None) == []
    assert service.get_insights("unknown-user") == []
    with service.repo._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_insights").fetchone()[0] == 0


def test_user_delete_removes_insights(registered_service):
    service, _ = registered_service
    service.apply_extracted("u", ExtractedPreferences(items=[
        PreferenceItem(category="interest", value="museums")
    ]))

    assert service.repo.delete("u") is True
    assert service.insight_repo.list("u") == []


def test_raw_evidence_lives_only_in_raw_text_and_is_deleted_by_forget(
    registered_service,
):
    service, db_path = registered_service
    evidence = "I strongly prefer museums."
    service.apply_extracted("u", ExtractedPreferences(insights=[
        UserInsight(
            insight_type="interest",
            key="museums",
            value="museums",
            raw_text=evidence,
            embedding=[0.1, 0.2],
        )
    ]))

    record = service.get_insights("u")[0]
    assert record["raw_text"] == evidence
    assert all(
        field != evidence
        for key, field in record.items()
        if key != "raw_text"
    )
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(user_insights)")}
    assert "embedding" not in columns

    service.forget("u", "interest", "museum")
    assert service.get_insights("u") == []


def test_source_has_no_in_memory_insights_store():
    from pathlib import Path

    assert not any(
        "USER_INSIGHTS_STORE" in source.read_text(encoding="utf-8")
        for source in Path("src").rglob("*.py")
    )
