"""
Unit and Integration Tests for User Authentication & SQLite Repository
Tests SQLiteUserRepository, Password Hashing, JWT Tokens, AuthService,
and FastAPI Authentication Endpoints.
Module: tests/test_auth.py
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.db.repositories.user_repo import (
    UserRecord,
    SQLiteUserRepository,
)
from src.services.auth import (
    AuthService,
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    validate_username,
    validate_password,
    DEMO_PASSWORD,
)
from src.db.profile_store import ProfileStore
from src.services.profile_service import ProfileService
from src.main import app


@pytest.fixture
def temp_repo(tmp_path):
    """Provides a fresh, isolated SQLiteUserRepository per test."""
    db_file = str(tmp_path / "test_auth.db")
    return SQLiteUserRepository(db_path=db_file)


@pytest.fixture
def temp_auth_service(temp_repo):
    """Provides an AuthService backed by an isolated SQLite database."""
    return AuthService(user_repo=temp_repo)


# =========================================================================
# 1. SQLiteUserRepository Tests (Repository Pattern)
# =========================================================================

def test_sqlite_user_repo_create_and_fetch(temp_repo):
    """Tests user creation and retrieval by username and user_id."""
    record = UserRecord(
        user_id="layla_egypt",
        username="layla_egypt",
        name="Layla Ahmed",
        email="layla@example.com",
        hashed_password=hash_password("secret123"),
        nationality="Egyptian",
    )
    created = temp_repo.create(record)
    assert created.id is not None
    assert created.user_id == "layla_egypt"

    # Fetch by username (case-insensitive)
    fetched = temp_repo.get_by_username("LAYLA_EGYPT")
    assert fetched is not None
    assert fetched.name == "Layla Ahmed"
    assert fetched.email == "layla@example.com"

    # Fetch by user_id
    by_id = temp_repo.get_by_user_id("layla_egypt")
    assert by_id is not None
    assert by_id.name == "Layla Ahmed"

    # Fetch by email
    by_email = temp_repo.get_by_email("LAYLA@example.com")
    assert by_email is not None
    assert by_email.user_id == "layla_egypt"


def test_sqlite_user_repo_duplicate_username_fails(temp_repo):
    """Ensures duplicate usernames are rejected by SQLite unique constraint."""
    record1 = UserRecord(
        user_id="user_dup",
        username="user_dup",
        name="User One",
        hashed_password=hash_password("secret123"),
    )
    temp_repo.create(record1)

    record2 = UserRecord(
        user_id="user_dup",
        username="user_dup",
        name="User Two",
        hashed_password=hash_password("secret456"),
    )
    with pytest.raises(Exception):
        temp_repo.create(record2)


def test_sqlite_user_repo_list_and_delete(temp_repo):
    """Tests listing all users and deleting by user_id."""
    temp_repo.create(UserRecord(
        user_id="u1", username="u1", name="One", hashed_password="h1"
    ))
    temp_repo.create(UserRecord(
        user_id="u2", username="u2", name="Two", hashed_password="h2"
    ))

    all_users = temp_repo.list_all()
    assert len(all_users) == 2

    deleted = temp_repo.delete("u1")
    assert deleted is True
    assert temp_repo.get_by_user_id("u1") is None
    assert len(temp_repo.list_all()) == 1


# =========================================================================
# 2. Security, Password Hashing & JWT Tests
# =========================================================================

def test_password_hashing_and_verification():
    """Tests bcrypt password hashing and verification."""
    pwd = "MySecretPassword2026!"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False


def test_jwt_token_encoding_and_decoding():
    """Tests JWT access token creation and decoding."""
    payload = {"sub": "omar_trav", "name": "Omar Ali", "nationality": "Egyptian"}
    token = create_access_token(payload)
    assert isinstance(token, str)

    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "omar_trav"
    assert decoded["name"] == "Omar Ali"


def test_validation_helpers():
    """Tests username and password validation rules."""
    assert validate_username("valid_user123")[0] is True
    assert validate_username("ab")[0] is False  # too short
    assert validate_username("invalid space")[0] is False  # spaces
    assert validate_username("user@invalid")[0] is False  # special chars

    assert validate_password("123456")[0] is True
    assert validate_password("12345")[0] is False  # too short
    assert validate_password("")[0] is False


# =========================================================================
# 3. AuthService Tests (Registration & Login)
# =========================================================================

def test_auth_service_registration_success(temp_auth_service):
    """Tests successful user registration via AuthService."""
    success, msg, user = temp_auth_service.register(
        username="tariq_cairo",
        password="securePassword123",
        name="Tariq Cairo",
        email="tariq@ra7al.eg",
        nationality="Egyptian",
    )
    assert success is True
    assert user is not None
    assert user.username == "tariq_cairo"
    assert user.name == "Tariq Cairo"

    # Profile provisioned in SQLite / ProfileService
    profile = ProfileService(temp_auth_service.user_repo).get_or_init("tariq_cairo")
    assert profile is not None
    assert profile.name == "Tariq Cairo"


def test_auth_service_registration_duplicate_fails(temp_auth_service):
    """Tests that registering an existing username returns failure."""
    temp_auth_service.register(
        username="unique_user",
        password="password123",
        name="Unique Name",
    )
    success, msg, user = temp_auth_service.register(
        username="unique_user",
        password="otherPassword123",
        name="Another Name",
    )
    assert success is False
    assert "already taken" in msg
    assert user is None


def test_auth_service_login_success_and_failure(temp_auth_service):
    """Tests user login with correct and incorrect credentials."""
    temp_auth_service.register(
        username="khalid_alex",
        password="khalidPassword2026",
        name="Khalid Alexandria",
    )

    # Correct credentials
    ok, msg, user, token = temp_auth_service.login("khalid_alex", "khalidPassword2026")
    assert ok is True
    assert user is not None
    assert token is not None
    assert user.username == "khalid_alex"

    # Incorrect password
    bad_ok, bad_msg, bad_user, bad_token = temp_auth_service.login("khalid_alex", "wrongPassword")
    assert bad_ok is False
    assert bad_user is None
    assert bad_token is None

    # Non-existent user
    no_ok, no_msg, _, _ = temp_auth_service.login("ghost_user", "anyPassword")
    assert no_ok is False


def test_auth_service_demo_users_seeded(temp_auth_service):
    """Tests that default demo users are automatically seeded and authenticatable."""
    ok, msg, user, token = temp_auth_service.login("user_101", DEMO_PASSWORD)
    assert ok is True
    assert user is not None
    assert user.user_id == "user_101"


# =========================================================================
# 4. ProfileStore Integration with SQLite
# =========================================================================

def test_profile_store_retrieves_sqlite_registered_user(temp_repo):
    """Ensures ProfileStore can resolve users created in SQLite repository."""
    ProfileStore.user_repo = temp_repo
    try:
        # Register user directly in SQLite
        temp_repo.create(UserRecord(
            user_id="sqlite_explorer",
            username="sqlite_explorer",
            name="SQLite Explorer",
            hashed_password=hash_password("pass123"),
            nationality="Egyptian",
        ))

        # ProfileStore should hydrate UserPreferences
        profile = ProfileStore.get_user_profile("sqlite_explorer")
        assert profile is not None
        assert profile.user_id == "sqlite_explorer"
        assert profile.name == "SQLite Explorer"
    finally:
        ProfileStore.user_repo = None


# =========================================================================
# 5. FastAPI Auth Endpoints Tests
# =========================================================================

def test_api_register_and_login_flow():
    """Tests full API registration, login, and /me retrieval."""
    import uuid
    client = TestClient(app)
    unique_user = f"api_user_{uuid.uuid4().hex[:8]}"

    # 1. Register
    reg_payload = {
        "username": unique_user,
        "password": "strongPassword123",
        "name": "API Traveler",
        "email": f"{unique_user}@example.com",
        "nationality": "Egyptian",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["username"] == unique_user
    assert "access_token" in reg_data
    token = reg_data["access_token"]

    # 2. Login
    login_payload = {
        "username": unique_user,
        "password": "strongPassword123",
    }
    login_res = client.post("/api/v1/auth/login", json=login_payload)
    assert login_res.status_code == 200
    login_data = login_res.json()
    assert login_data["access_token"] is not None

    # 3. Login with wrong password -> 401
    bad_login = client.post("/api/v1/auth/login", json={"username": unique_user, "password": "wrong"})
    assert bad_login.status_code == 401

    # 4. Access /me with valid token
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == unique_user
    assert me_data["name"] == "API Traveler"

    # 5. Access /me without token -> 401
    me_unauth = client.get("/api/v1/auth/me")
    assert me_unauth.status_code == 401
