# Location resolution

`reverse_geocode(lat, lon)` returns a `LocationResponse` with structured address fields, original coordinates, a short `label`, `source`, and a stable short error code. `formatted_address` is optional and is never populated with a failure message.

The resolver records `state["location_details"]` with:

- `status`: `"ok"` after successful resolution or `"unavailable"` after a failed lookup.
- `stale`: `true` only when a failed lookup leaves a previously known `current_location` in place.
- `lat`, `lon`, `city`, `governorate`, `neighborhood`, `country`, `country_code`, `in_egypt`, `formatted_address`, `label`, `error`, and `source`.

On failure, raw coordinates are preserved for a later retry and `current_location` is left untouched; it remains absent when no previous location was known. On success, `current_location` uses the short label (for example, `Zamalek, Cairo`), raw coordinates are cleared, and coordinates remain available in `location_details`.

Providers are configured with `GEOCODER_PROVIDER`: `nominatim` (default), `mock` (deterministic Zamalek/Cairo result), or `mock_fail` (stable `mock_unavailable` result for tests). Nominatim lookups use an in-process, bounded cache keyed by coordinates rounded to three decimal places. Successful and failed lookups use separate TTLs, and uncached public-provider requests are throttled.

The public Nominatim server requires an identifying User-Agent; configure `GEOCODER_CONTACT` with contact information. It allows about one request per second and is not intended for heavy use. Use a self-hosted instance or commercial provider in production.
