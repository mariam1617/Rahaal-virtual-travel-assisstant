from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator
from src.state.schema import UserPreferences, TripRequirements, GeneratedPlan

class OverallState(TypedDict):
    # مدخلات ومحادثة
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_id: str
    
    # الذاكرة الدائمة (Long-Term Memory)
    user_preferences: Optional[UserPreferences]
    
    # بيانات الرحلة الحالية (Short-Term Session)
    trip_requirements: Optional[TripRequirements]
    
    # مخرجات ومسودة الخطط التي تتنقل بين Planner / Validator / Supervisor
    current_plan: Optional[GeneratedPlan]
    
    # متغيرات التوجيه والتنفيذ
    next_agent: Optional[str]
    errors: Annotated[List[str], operator.add]
    booking_status: Optional[str]