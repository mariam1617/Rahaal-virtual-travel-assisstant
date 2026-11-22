"""
Authentication Domain Service
Handles password hashing, token issuance, credential validation,
and user registration/login via the Repository Pattern.
Module: src/services/auth.py
"""

from datetime import datetime, timedelta, timezone
import os
import re
from typing import Dict, Any, Optional, Tuple
import bcrypt
import jwt

from src.db.repositories.user_repo import (
    IUserRepository,
    SQLiteUserRepository,
    UserRecord,
)
from src.state.memory import MOCK_USER_DB
from src.state.schema import DietaryPreference


# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ra7al-egypt-travel-concierge-secret-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Demo Password for legacy demo accounts
DEMO_PASSWORD = os.getenv("RA7AL_DEMO_PASSWORD", "ra7al2026")


def hash_password(plain_password: str) -> str:
    """Hashes a password using bcrypt with a generated salt."""
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Generates a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """Enforces username standards: 3-30 chars, alphanumeric and underscores."""
    clean = (username or "").strip()
    if len(clean) < 3:
        return False, "Username must be at least 3 characters long."
    if len(clean) > 30:
        return False, "Username cannot exceed 30 characters."
    if not re.match(r"^[a-zA-Z0-9_]+$", clean):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, None


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """Enforces minimum password length of 6 characters."""
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, None


class AuthService:
    """
    Coordinates user registration, credential authentication,
    and profile hydration across the repository.
    """

    def __init__(self, user_repo: Optional[IUserRepository] = None):
        self.user_repo: IUserRepository = user_repo or SQLiteUserRepository()
        self._ensure_demo_users_seeded()

    def _ensure_demo_users_seeded(self) -> None:
        """Seeds default demo accounts if not already present in the repository."""
        demo_accounts = [
            ("test_user", "Test Traveler", "Egyptian"),
            ("user_101", "Traveler 101", "Egyptian"),
            ("user_202", "Traveler 202", "German"),
        ]
        demo_hash = hash_password(DEMO_PASSWORD)

        for username, display_name, nat in demo_accounts:
            try:
                if not self.user_repo.get_by_username(username):
                    record = UserRecord(
                        user_id=username,
                        username=username,
                        name=display_name,
                        email=f"{username}@ra7al.demo",
                        hashed_password=demo_hash,
                        nationality=nat,
                    )
                    self.user_repo.create(record)
            except Exception:
                # User already exists or database locked
                pass

    def register(
        self,
        username: str,
        password: str,
        name: str,
        email: Optional[str] = None,
        nationality: str = "Egyptian"
    ) -> Tuple[bool, str, Optional[UserRecord]]:
        """
        Registers a new traveler account.
        Returns: (success, message, user_record)
        """
        # 1. Validate credentials
        valid_user, user_err = validate_username(username)
        if not valid_user:
            return False, user_err or "Invalid username.", None

        valid_pwd, pwd_err = validate_password(password)
        if not valid_pwd:
            return False, pwd_err or "Invalid password.", None

        display_name = (name or "").strip()
        if not display_name:
            return False, "Full Name cannot be empty.", None

        clean_username = username.strip().lower()

        # 2. Check uniqueness
        if self.user_repo.get_by_username(clean_username):
            return False, f"Username '{clean_username}' is already taken.", None

        clean_email = email.strip().lower() if email else None
        if clean_email and self.user_repo.get_by_email(clean_email):
            return False, f"Email '{clean_email}' is already registered.", None

        # 3. Hash password and persist
        hashed = hash_password(password)
        user_record = UserRecord(
            user_id=clean_username,
            username=clean_username,
            name=display_name,
            email=clean_email,
            hashed_password=hashed,
            nationality=nationality.strip() if nationality else "Egyptian",
        )

        try:
            persisted_user = self.user_repo.create(user_record)

            # 4. Provision Traveler Profile for Concierge Agents
            MOCK_USER_DB[clean_username] = {
                "user_id": clean_username,
                "name": display_name,
                "nationality": user_record.nationality,
                "dietary": [DietaryPreference.NONE],
                "budget_level": "moderate",
                "interests": [],
                "accessibility_needs": [],
                "travel_habits": [],
            }

            return True, "Account created successfully.", persisted_user
        except Exception as e:
            return False, f"Registration failed: {str(e)}", None

    def login(
        self,
        username: str,
        password: str
    ) -> Tuple[bool, str, Optional[UserRecord], Optional[str]]:
        """
        Authenticates a user by username and password.
        Returns: (success, message, user_record, access_token)
        """
        clean_user = (username or "").strip().lower()
        if not clean_user or not password:
            return False, "Username and password are required.", None, None

        user = self.user_repo.get_by_username(clean_user)
        if not user:
            return False, "Incorrect username or password.", None, None

        if not verify_password(password, user.hashed_password):
            return False, "Incorrect username or password.", None, None

        # Issue JWT Access Token
        token = create_access_token({
            "sub": user.user_id,
            "username": user.username,
            "name": user.name,
            "nationality": user.nationality,
        })

        return True, "Authentication successful.", user, token

    def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Retrieves a user by user_id."""
        return self.user_repo.get_by_user_id(user_id)
