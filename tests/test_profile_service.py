import sqlite3
import pytest

from src.db.repositories.user_repo import SQLiteUserRepository, UserRecord
from src.services.profile_service import ProfileService, UnknownUserError, merge_extracted
from src.state.schema import DietaryPreference, ExtractedPreferences, PreferenceItem, UserPreferences
from src.agents.feedback.preference_learning import ltm_preference_update_node


@pytest.fixture
def svc(tmp_path):
    repo = SQLiteUserRepository(db_path=str(tmp_path / "t.db"))
    service = ProfileService(repo)
    ProfileService.set_default(service)
    yield service
    ProfileService.set_default(None)


def test_new_user_without_nationality_is_none(svc):
    user = svc.repo.create(UserRecord(
        user_id="user_nonat",
        username="user_nonat",
        name="No Nat",
        hashed_password="pw"
    ))
    profile = svc.get_or_init("user_nonat")
    assert profile is not None
    assert profile.nationality is None


def test_value_roundtrip(svc):
    svc.repo.create(UserRecord(
        user_id="user_german",
        username="user_german",
        name="Hans",
        nationality="German",
        hashed_password="pw"
    ))
    pref = svc.get_or_init("user_german")
    assert pref.nationality == "German"

    extracted = ExtractedPreferences(
        dietary=[DietaryPreference.VEGAN, DietaryPreference.HALAL],
        interests=["castles", "hiking"],
        travel_habits=["early morning flights"],
    )
    updated = svc.apply_extracted("user_german", extracted)
    assert updated.nationality == "German"
    assert DietaryPreference.VEGAN in updated.dietary
    assert DietaryPreference.HALAL in updated.dietary
    assert "castles" in updated.liked_interests
    assert "early morning flights" in updated.travel_habits

    def add_avoided(d):
        d["avoided_interests"] = ["crowds", "nightclubs"]
        return d

    svc.repo.update_profile_atomic("user_german", add_avoided)
    reloaded = svc.get("user_german")
    assert reloaded is not None
    assert reloaded.nationality == "German"
    assert reloaded.dietary == [DietaryPreference.VEGAN, DietaryPreference.HALAL]
    assert reloaded.travel_habits == ["early morning flights"]
    assert reloaded.avoided_interests == ["crowds", "nightclubs"]
    assert "castles" in reloaded.liked_interests


def test_legacy_egyptian_row_reads_egyptian(svc):
    with svc.repo._connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, username, name, hashed_password, nationality) VALUES (?, ?, ?, ?, ?)",
            ("legacy_user", "legacy_user", "Legacy User", "pw", "Egyptian")
        )
        conn.execute(
            "INSERT INTO user_profiles (user_id, name, nationality) VALUES (?, ?, ?)",
            ("legacy_user", "Legacy User", "Egyptian")
        )
        conn.commit()

    profile = svc.get("legacy_user")
    assert profile is not None
    assert profile.nationality == "Egyptian"


def test_migration_adds_avoided_interests_column(tmp_path):
    old_db = str(tmp_path / "old.db")
    conn = sqlite3.connect(old_db)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            hashed_password TEXT NOT NULL,
            nationality TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            nationality TEXT,
            dietary_json TEXT DEFAULT '[]',
            budget_level TEXT DEFAULT 'moderate',
            interests_json TEXT DEFAULT '["museums"]',
            accessibility_needs_json TEXT DEFAULT '[]',
            travel_habits_json TEXT DEFAULT '[]',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("INSERT INTO users (user_id, username, name, hashed_password) VALUES ('u1', 'u1', 'U1', 'pw')")
    conn.execute("INSERT INTO user_profiles (user_id, name, interests_json) VALUES ('u1', 'U1', '[\"museums\"]')")
    conn.commit()
    conn.close()

    repo = SQLiteUserRepository(db_path=old_db)
    svc = ProfileService(repo)
    p = svc.get("u1")
    assert p is not None
    assert p.liked_interests == ["museums"]
    assert p.avoided_interests == []


def test_get_no_side_effects_and_unknown_id(svc):
    with svc.repo._connect() as conn:
        initial_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]

    res = svc.get("unknown_user_123")
    assert res is None

    with svc.repo._connect() as conn:
        count_after_get = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
    assert count_after_get == initial_count

    res_init = svc.get_or_init("unknown_user_123")
    assert res_init is None
    with svc.repo._connect() as conn:
        count_after_init = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
    assert count_after_init == initial_count

    svc.repo.create(UserRecord(
        user_id="reg_user",
        username="reg_user",
        name="Reg User",
        hashed_password="pw"
    ))
    res_reg = svc.get_or_init("reg_user")
    assert res_reg is not None
    assert res_reg.user_id == "reg_user"
    with svc.repo._connect() as conn:
        count_after_reg = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
    assert count_after_reg == initial_count + 1


def test_guest_isolation(svc):
    with svc.repo._connect() as conn:
        initial_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]

    ex = ExtractedPreferences(interests=["diving"], travel_habits=["night flights"])
    g1 = svc.apply_extracted(None, ex)
    assert g1.user_id == "guest"
    assert "diving" in g1.liked_interests

    g2 = svc.apply_extracted("not-a-user", ex)
    assert g2.user_id == "guest"

    with svc.repo._connect() as conn:
        final_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
    assert final_count == initial_count

    fresh1 = svc.guest_profile()
    fresh2 = svc.guest_profile()
    assert fresh1 is not fresh2
    fresh1.liked_interests.append("unique_to_1")
    assert "unique_to_1" not in fresh2.liked_interests


def test_ltm_preference_update_guest_isolation(svc):
    with svc.repo._connect() as conn:
        initial_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        test_user_row = conn.execute("SELECT * FROM users WHERE user_id = 'test_user'").fetchone()
        assert test_user_row is None

    state = {
        "messages": [("user", "I prefer morning flights and eat vegan")],
        "feedback_type": "long_term_preference"
    }
    res = ltm_preference_update_node(state)
    assert res["user_preferences"] is not None
    assert DietaryPreference.VEGAN in res["user_preferences"].dietary
    assert "morning flights" in res["user_preferences"].travel_habits
    assert "session" in res["response"].lower()

    with svc.repo._connect() as conn:
        final_count = conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        test_user_row = conn.execute("SELECT * FROM users WHERE user_id = 'test_user'").fetchone()
    assert final_count == initial_count
    assert test_user_row is None


def test_repo_delete_removes_profile(svc):
    svc.repo.create(UserRecord(
        user_id="del_user",
        username="del_user",
        name="Del User",
        hashed_password="pw"
    ))
    svc.get_or_init("del_user")
    assert svc.get("del_user") is not None

    deleted = svc.repo.delete("del_user")
    assert deleted is True
    assert svc.repo.get_by_user_id("del_user") is None
    assert svc.repo.get_profile("del_user") is None


def test_merge_extracted_parity():
    p = UserPreferences(
        user_id="u",
        name="U",
        dietary=[DietaryPreference.NONE],
        budget_level="budget",
        liked_interests=["pyramids"],
    )
    ex = ExtractedPreferences(
        dietary=[DietaryPreference.VEGAN],
        interests=["museums", "pyramids"],
        budget_level="luxury",
        travel_habits=["early start"],
    )
    merged = merge_extracted(p, ex)
    assert DietaryPreference.NONE not in merged.dietary
    assert DietaryPreference.VEGAN in merged.dietary
    assert sorted(merged.liked_interests) == ["museums", "pyramids"]
    assert merged.budget_level == "luxury"
    assert merged.travel_habits == ["early start"]


def test_dislike_museums_moves_to_avoided():
    p = UserPreferences(user_id="u", name="U", liked_interests=["museums"])
    ex = ExtractedPreferences(items=[PreferenceItem(category="interest", value="museums", polarity="avoid")])
    out = merge_extracted(p, ex)
    assert "museums" not in out.liked_interests and "museums" in out.avoided_interests
