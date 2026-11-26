from typing import Literal, Optional
from pydantic import BaseModel

class LocationResponse(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    formatted_address: Optional[str] = None
    error: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    governorate: Optional[str] = None
    neighborhood: Optional[str] = None
    country_code: Optional[str] = None
    source: Optional[Literal["nominatim", "cache", "mock"]] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def in_egypt(self) -> Optional[bool]:
        return None if not self.country_code else self.country_code == "eg"

    @property
    def label(self) -> Optional[str]:
        parts = list(dict.fromkeys(
            part for part in (self.neighborhood, self.city or self.governorate) if part
        ))
        if parts:
            return ", ".join(parts)
        if self.formatted_address:
            return ", ".join(
                part.strip() for part in self.formatted_address.split(",")[:2] if part.strip()
            )
        return None

    def to_details(self, status: str, stale: bool = False) -> dict:
        return {
            "status": status,
            "stale": stale,
            "lat": self.lat,
            "lon": self.lon,
            "city": self.city,
            "governorate": self.governorate,
            "neighborhood": self.neighborhood,
            "country": self.country,
            "country_code": self.country_code,
            "in_egypt": self.in_egypt,
            "formatted_address": self.formatted_address,
            "label": self.label,
            "error": self.error,
            "source": self.source,
        }