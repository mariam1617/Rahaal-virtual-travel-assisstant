from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
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
    # TODO (Problem 1): Unknown nationality currently defaults to a fixed string ("Egyptian" or "unknown").
    # Make it Optional[str] = None instead of assuming Egyptian nationality. Once visa advice uses
    # user_preferences.nationality (plan step 2.5), a foreign tourist with no stated nationality would
    # incorrectly be told they need no visa. It also affects foreigner vs Egyptian ticket pricing tiers.
    # Ask when a nationality-dependent question comes up.
    nationality: Optional[str] = None
    dietary: List[DietaryPreference] = Field(default_factory=list)
    budget_level: str = "moderate"  # budget, moderate, luxury
    # TODO (Problem 3): Split interests into like/avoid categories (e.g. liked_interests vs avoided_interests)
    # and add travel_habits: List[str] = Field(default_factory=list). Currently travel_habits are saved
    # to the DB but UserPreferences drops them on every read because the field is missing.
    interests: List[str] = Field(default_factory=list) # e.g., ["historical", "nature", "cafes"]
    accessibility_needs: Optional[List[str]] = Field(default_factory=list)

# --- 2. بيانات الرحلة الحالية المؤقتة (Current Trip / Session Data) ---
class TripRequirements(BaseModel):
    destination: str
    duration_days: Optional[int] = None
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

# --- 3. بيانات التغذية الراجعة والذاكرة طويلة المدى (Feedback & Learning) ---
FeedbackType = Literal["trip_adjustment", "long_term_preference", "both", "unknown"]

class FeedbackClassification(BaseModel):
    feedback_type: FeedbackType = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    modification_intent: Optional[str] = None
    preference_intent: Optional[str] = None

class UserInsight(BaseModel):
    insight_type: str  # dietary, travel_habit, budget_level, interest, accessibility
    key: str
    value: Any
    tags: List[str] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float = 1.0
    embedding: Optional[List[float]] = None

class ExtractedPreferences(BaseModel):
    dietary: List[DietaryPreference] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    budget_level: Optional[str] = None
    accessibility_needs: List[str] = Field(default_factory=list)
    travel_habits: List[str] = Field(default_factory=list)
    insights: List[UserInsight] = Field(default_factory=list)

class PlanModification(BaseModel):
    action: Literal["replace", "remove", "add", "reschedule", "general"] = "replace"
    day_number: Optional[int] = 1
    time_slot: Optional[str] = None  # morning, afternoon, evening
    target_place_or_activity: Optional[str] = None
    replacement_name: Optional[str] = None
    replacement_category: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    notes: Optional[str] = None

