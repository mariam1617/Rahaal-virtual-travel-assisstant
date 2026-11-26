
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlannerRequest(BaseModel):
    destination: str
    duration_days: int = Field(default=3, ge=1, le=14)
    total_budget: float = Field(default=500.0, gt=0)
    accessibility_needs: List[str] = Field(default_factory=list)


class PlannerResponse(BaseModel):
    plan: Dict[str, Any]
    is_constraint_valid: bool
    is_advanced_valid: bool
    is_executable: bool
    evaluation: Dict[str, Any]
    response: str = ""
    temporal_errors: List[str] = Field(default_factory=list)
    accessibility_errors: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    approval_required: bool = False
    approval_status: str = "pending"
    booking_status: str = "not_started"
    booking_attempts: int = 0
    bookings: List[Dict[str, Any]] = Field(default_factory=list)
    booking_errors: List[str] = Field(default_factory=list)
    calendar_status: str = "not_started"
    calendar_events: List[Dict[str, Any]] = Field(default_factory=list)
    calendar_errors: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Authentication Schemas
# ----------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, description="Unique username identifier.")
    password: str = Field(..., min_length=6, description="Account password.")
    name: str = Field(..., min_length=1, description="Display or full name.")
    email: Optional[str] = Field(default=None, description="Optional contact email.")
    nationality: str = Field(default="Egyptian", description="Traveler nationality.")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Registered username.")
    password: str = Field(..., description="Account password.")


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    name: str
    nationality: str = "Egyptian"
    message: str = ""


class UserResponse(BaseModel):
    user_id: str
    username: str
    name: str
    email: Optional[str] = None
    nationality: str = "Egyptian"
    created_at: Optional[str] = None

