import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.config import settings
from src.services import location_services as geocoder
from src.services.location import LocationResponse


CAIRO_RESPONSE = {
    "display_name": "Zamalek, Cairo, Egypt",
    "address": {
        "suburb": "Zamalek",
        "city": "Cairo",
        "state": "Cairo Governorate",
        "country": "Egypt",
        "country_code": "eg",
    },
}


class FakeResponse:
    def __init__(self, data=None, status_error=None):
        self.data = data
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


@pytest.fixture
def geocoder_setup(monkeypatch):
    clock = [100.0]
    sleeps = []
    with geocoder._cache_lock:
        geocoder._cache.clear()
    monkeypatch.setattr(geocoder, "_now", lambda: clock[0])
    monkeypatch.setattr(geocoder, "_last", 0.0)
    monkeypatch.setattr(geocoder.time, "sleep", lambda wait: sleeps.append(wait))
    monkeypatch.setattr(settings, "geocoder_provider", "nominatim")
    monkeypatch.setattr(settings, "geocoder_base_url", "https://geo.test/reverse")
    monkeypatch.setattr(settings, "geocoder_contact", "geo@example.test")
    monkeypatch.setattr(settings, "geocoder_timeout_s", 5.0)
    monkeypatch.setattr(settings, "geocoder_cache_ttl_s", 60)
    monkeypatch.setattr(settings, "geocoder_fail_ttl_s", 10)
    monkeypatch.setattr(settings, "geocoder_min_interval_s", 1.0)
    return clock, sleeps


def test_reverse_geocode_parses_structured_cairo_result(monkeypatch, geocoder_setup):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(CAIRO_RESPONSE)

    monkeypatch.setattr(geocoder.requests, "get", fake_get)
    result = geocoder.reverse_geocode(30.0609, 31.2197)

    assert result.label == "Zamalek, Cairo"
    assert result.governorate == "Cairo"
    assert result.in_egypt is True
    assert result.source == "nominatim"
    assert calls[0][0] == "https://geo.test/reverse"
    assert calls[0][1]["params"] == {
        "lat": 30.0609, "lon": 31.2197, "format": "jsonv2",
        "zoom": 16, "addressdetails": 1, "accept-language": "en",
    }
    assert calls[0][1]["headers"]["User-Agent"] == f"{settings.project_name} (geo@example.test)"


def test_cache_key_ttl_and_source(monkeypatch, geocoder_setup):
    clock, _ = geocoder_setup
    calls = []
    monkeypatch.setattr(
        geocoder.requests, "get",
        lambda *args, **kwargs: (calls.append(1) or FakeResponse(CAIRO_RESPONSE)),
    )

    first = geocoder.reverse_geocode(30.0609, 31.2197)
    second = geocoder.reverse_geocode(30.0611, 31.2198)
    assert len(calls) == 1
    assert second.source == "cache"
    assert first.source == "nominatim"

    clock[0] += settings.geocoder_cache_ttl_s + 1
    refreshed = geocoder.reverse_geocode(30.0609, 31.2197)
    assert len(calls) == 2
    assert refreshed.source == "nominatim"


def test_negative_cache_expires_and_retries(monkeypatch, geocoder_setup):
    clock, _ = geocoder_setup
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("sensitive connection detail")
        return FakeResponse(CAIRO_RESPONSE)

    monkeypatch.setattr(geocoder.requests, "get", fake_get)
    failed = geocoder.reverse_geocode(30.0609, 31.2197)
    cached = geocoder.reverse_geocode(30.0609, 31.2197)
    assert failed.error == "http_error"
    assert cached.source == "cache"
    assert len(calls) == 1

    clock[0] += settings.geocoder_fail_ttl_s + 1
    assert geocoder.reverse_geocode(30.0609, 31.2197).ok
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response,error,raw",
    [
        (requests.Timeout("sensitive timeout detail"), "timeout", "sensitive timeout detail"),
        (requests.ConnectionError("sensitive connection detail"), "http_error", "sensitive connection detail"),
        (FakeResponse(status_error=requests.HTTPError("sensitive status detail")), "http_error", "sensitive status detail"),
        (FakeResponse(ValueError("sensitive JSON detail")), "bad_response", "sensitive JSON detail"),
        (FakeResponse({"error": "Unable to geocode"}), "no_address", "Unable to geocode"),
    ],
)
def test_error_codes_do_not_expose_exception_text(
    monkeypatch, geocoder_setup, response, error, raw
):
    def fake_get(*args, **kwargs):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(geocoder.requests, "get", fake_get)
    result = geocoder.reverse_geocode(30.0609, 31.2197)

    assert result.error == error
    assert result.formatted_address is None
    assert raw not in result.model_dump_json()


def test_invalid_coordinates_skip_http(monkeypatch, geocoder_setup):
    calls = []
    monkeypatch.setattr(
        geocoder.requests, "get",
        lambda *args, **kwargs: calls.append(1),
    )

    result = geocoder.reverse_geocode(200, 31.2)

    assert result.error == "invalid_coordinates"
    assert calls == []


def test_throttle_waits_between_uncached_requests(monkeypatch, geocoder_setup):
    _, sleeps = geocoder_setup
    monkeypatch.setattr(geocoder.requests, "get", lambda *args, **kwargs: FakeResponse(CAIRO_RESPONSE))

    geocoder.reverse_geocode(30.0609, 31.2197)
    geocoder.reverse_geocode(30.0619, 31.2197)

    assert sleeps and sleeps[0] > 0


def test_non_egypt_result_and_mock_providers(monkeypatch, geocoder_setup):
    monkeypatch.setattr(geocoder.requests, "get", lambda *args, **kwargs: FakeResponse({
        "display_name": "Paris, France",
        "address": {"city": "Paris", "country": "France", "country_code": "FR"},
    }))
    assert geocoder.reverse_geocode(48.8566, 2.3522).in_egypt is False

    monkeypatch.setattr(settings, "geocoder_provider", "mock")
    mock = geocoder.reverse_geocode(30.0444, 31.2357)
    assert mock.source == "mock"
    assert mock.label == "Zamalek, Cairo"

    monkeypatch.setattr(settings, "geocoder_provider", "mock_fail")
    assert geocoder.reverse_geocode(30.0444, 31.2357).error == "mock_unavailable"


def test_resolve_location_node_failure_and_success(monkeypatch):
    from src.agents.tourist_help import intent_resolver

    monkeypatch.setattr(
        intent_resolver,
        "reverse_geocode",
        lambda **kwargs: LocationResponse(error="timeout", lat=kwargs["lat"], lon=kwargs["lon"]),
    )
    old_location = {"raw_coordinates": {"lat": 30.04, "lon": 31.23}}
    failed = intent_resolver.resolve_location_node(old_location)
    assert "current_location" not in failed
    assert failed["raw_coordinates"] == old_location["raw_coordinates"]
    assert failed["location_details"]["status"] == "unavailable"
    assert failed["location_details"]["stale"] is False
    assert "Location unavailable" not in json.dumps(failed, default=str)

    previous = {**old_location, "current_location": "Old label"}
    stale = intent_resolver.resolve_location_node(previous)
    assert stale["current_location"] == "Old label"
    assert stale["location_details"]["stale"] is True

    monkeypatch.setattr(
        intent_resolver,
        "reverse_geocode",
        lambda **kwargs: LocationResponse(
            city="Cairo", neighborhood="Zamalek", country="Egypt",
            country_code="eg", lat=kwargs["lat"], lon=kwargs["lon"],
        ),
    )
    success = intent_resolver.resolve_location_node(old_location)
    assert success["current_location"] == "Zamalek, Cairo"
    assert success["raw_coordinates"] is None
    assert success["location_details"]["lat"] == 30.04
    assert success["location_details"]["lon"] == 31.23


def test_mock_failure_graph_emergency_output_contains_no_unavailable_label(
    monkeypatch, geocoder_setup
):
    from src.agents.tourist_help import graph as help_graph

    monkeypatch.setattr(settings, "geocoder_provider", "mock_fail")
    monkeypatch.setattr(help_graph, "help_verification_node", lambda state: {
        "response": state.get("candidate_response"),
        "verified_response": state.get("candidate_response"),
        "is_verified": True,
    })
    fake_answer_module = SimpleNamespace(
        answer_emergency_query=lambda **kwargs: SimpleNamespace(
            context="", answer="Emergency assistance is ready.", sources=[], status="answered"
        )
    )
    monkeypatch.setitem(sys.modules, "src.rag.tourist_rag.answer", fake_answer_module)

    result = help_graph.create_tourist_help_graph().invoke({
        "messages": [("user", "I need an ambulance, emergency")],
        "raw_coordinates": {"lat": 30.0444, "lon": 31.2357},
    })

    assert result["response"]
    assert "Location unavailable" not in json.dumps(result, default=str)


def test_source_has_no_unavailable_location_literal():
    source_files = Path("src").rglob("*.py")
    assert not any(
        "Location unavailable" in source.read_text(encoding="utf-8")
        for source in source_files
    )
