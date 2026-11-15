from pydantic import BaseModel
from typing import Optional

class LocationResponse(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    formatted_address: str
    error: Optional[str] = None