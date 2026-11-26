import logging
import math
import re
import threading
import time
from collections import OrderedDict
from typing import Optional

import requests

from src.config import settings
from src.services.location import LocationResponse

logger = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[float, float], tuple[float, LocationResponse]] = OrderedDict()
_CACHE_MAX_SIZE = 512
_now = time.monotonic

_throttle_lock, _last = threading.Lock(), 0.0


def _throttle(min_interval: float) -> None:
    global _last
    with _throttle_lock:
        wait = min_interval - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()


def _cached(key: tuple[float, float]) -> Optional[LocationResponse]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, response = entry
        if _now() >= expires_at:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return response.model_copy(update={"source": "cache"})


def _cache_response(key: tuple[float, float], response: LocationResponse) -> None:
    ttl = settings.geocoder_fail_ttl_s if response.error else settings.geocoder_cache_ttl_s
    if ttl <= 0:
        return
    with _cache_lock:
        _cache[key] = (_now() + ttl, response.model_copy(deep=True))
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)


def _failure(lat: float, lon: float, error: str, source: Optional[str] = None) -> LocationResponse:
    return LocationResponse(lat=lat, lon=lon, error=error, source=source)


def _mock_location(lat: float, lon: float) -> LocationResponse:
    return LocationResponse(
        city="Cairo",
        country="Egypt",
        formatted_address="Zamalek, Cairo, Egypt",
        lat=lat,
        lon=lon,
        governorate="Cairo",
        neighborhood="Zamalek",
        country_code="eg",
        source="mock",
    )


def reverse_geocode(lat: float, lon: float) -> LocationResponse:
    """Resolve coordinates through the configured provider and a bounded local cache."""
    try:
        valid = (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return _failure(lat, lon, "invalid_coordinates")

    provider = settings.geocoder_provider.strip().lower()
    if provider == "mock":
        return _mock_location(lat, lon)
    if provider == "mock_fail":
        return _failure(lat, lon, "mock_unavailable", source="mock")
    if provider != "nominatim":
        logger.warning("Unsupported geocoder provider configured")
        return _failure(lat, lon, "disabled")

    key = (round(lat, 3), round(lon, 3))
    cached = _cached(key)
    if cached is not None:
        return cached

    _throttle(settings.geocoder_min_interval_s)
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": 16,
        "addressdetails": 1,
        "accept-language": "en",
    }
    headers = {
        "User-Agent": f"{settings.project_name} ({settings.geocoder_contact or 'no-contact'})"
    }

    try:
        response = requests.get(
            settings.geocoder_base_url,
            params=params,
            headers=headers,
            timeout=settings.geocoder_timeout_s,
        )
        response.raise_for_status()
    except requests.Timeout:
        logger.warning("Reverse geocoding timed out", exc_info=True)
        result = _failure(lat, lon, "timeout")
        _cache_response(key, result)
        return result
    except requests.RequestException:
        logger.warning("Reverse geocoding request failed", exc_info=True)
        result = _failure(lat, lon, "http_error")
        _cache_response(key, result)
        return result

    try:
        data = response.json()
    except ValueError:
        logger.warning("Reverse geocoding returned invalid JSON", exc_info=True)
        result = _failure(lat, lon, "bad_response")
        _cache_response(key, result)
        return result

    if not isinstance(data, dict):
        result = _failure(lat, lon, "bad_response")
    elif data.get("error"):
        result = _failure(lat, lon, "no_address")
    elif not isinstance(data.get("address", {}), dict):
        result = _failure(lat, lon, "bad_response")
    elif not data.get("display_name"):
        result = _failure(lat, lon, "no_address")
    else:
        address = data.get("address", {})
        state = address.get("state")
        governorate = re.sub(r"\s+Governorate$", "", state, flags=re.IGNORECASE) if state else None
        country_code = address.get("country_code")
        result = LocationResponse(
            city=address.get("city") or address.get("town") or address.get("village"),
            country=address.get("country"),
            formatted_address=data["display_name"],
            lat=lat,
            lon=lon,
            governorate=governorate,
            neighborhood=(
                address.get("suburb")
                or address.get("neighbourhood")
                or address.get("city_district")
                or address.get("quarter")
            ),
            country_code=country_code.lower() if isinstance(country_code, str) else None,
            source="nominatim",
        )

    _cache_response(key, result)
    return result
