from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# --- 1. تفضيلات المستخدم الدائمة (Long-Term Preferences) ---
class DietaryPreference(str, Enum):
    HALAL = "halal"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    NONE = "none"

class UserPreferences(BaseModel):
    user_id: str
    name: str
    nationality: str
    dietary: List[DietaryPreference] = Field(default_factory=list)
    budget_level: str = "moderate"  # budget, moderate, luxury
    interests: List[str] = Field(default_factory=list) # e.g., ["historical", "nature", "cafes"]
    accessibility_needs: Optional[List[str]] = Field(default_factory=list)

# --- 2. بيانات الرحلة الحالية المؤقتة (Current Trip / Session Data) ---
class TripRequirements(BaseModel):
    destination: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_budget: Optional[float] = None
    companions_count: int = 1
    current_location: Optional[Dict[str, float]] = None # {"lat": 30.0444, "lng": 31.2357}

class ItineraryDay(BaseModel):
    day_number: int
    activities: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_cost: float = 0.0

class GeneratedPlan(BaseModel):
    plan_id: Optional[str] = None
    title: Optional[str] = None
    days: List[ItineraryDay] = Field(default_factory=list)
    status: str = "draft"  # draft, validated, approved, booked