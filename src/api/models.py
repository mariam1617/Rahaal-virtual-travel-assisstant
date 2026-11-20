
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
