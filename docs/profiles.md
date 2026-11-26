# ProfileService & Preferences Reference

`ProfileService` (`src/services/profile_service.py`) is the unified service for traveler profiles and preference management.

### ProfileService API
- `ProfileService.default()`: Singleton instance backed by `SQLiteUserRepository`.
- `get(user_id) -> Optional[UserPreferences]`: Pure read with zero side effects. Returns `None` if no profile exists.
- `get_or_init(user_id) -> Optional[UserPreferences]`: Retrieves existing profile or initializes a default profile only for registered users (`users` table). Returns `None` for unregistered/guest users.
- `apply_extracted(user_id, extracted, current=None) -> UserPreferences`: Atomically persists updates for registered users via SQLite. For guests/unregistered users, applies pure `merge_extracted` in-memory without DB access.
- `apply_extracted_with_changes(user_id, extracted, current=None) -> ApplyResult`: Same update contract, returning the resulting profile and a human-readable list of effective changes.
- `forget(user_id, category, value) -> Optional[UserPreferences]`: Removes a registered user's matching preference; interest removal checks both liked and avoided interests, and an empty dietary list returns to `[NONE]`.
- `is_registered(user_id) -> bool`: Checks existence in the `users` table.
- `guest_profile() -> UserPreferences`: Returns a fresh ephemeral `UserPreferences(user_id="guest", name="Guest")`.

### Preference polarity
Long-term `PreferenceItem` values use `polarity="like"` or `"avoid"`. Interests move between `liked_interests` and `avoided_interests`; dietary, accessibility, and travel-habit values are added or removed; budget level is set only by a supported `like` item. Items scoped to `this_trip` are ignored by profile persistence and should instead be used by the current-trip modifier. Applying the same item repeatedly is idempotent.

`merge_extracted` and `merge_extracted_with_changes` are pure functions. The latter returns `ApplyResult(profile, changes)` and includes effective legacy-field additions as well as item changes.

### Guest Isolation Rule
- Guests and anonymous users are never persisted to SQLite (`users` or `user_profiles`).
- Ephemeral guest preferences live exclusively in `state['user_preferences']` for the current session.
- Guest and unknown-user updates, including preference items, are not written to SQLite.

### Schema & Field Changes
- `liked_interests`: Replaces legacy `interests` field (persisted in `interests_json` column).
- `interests`: Retained as a deprecated `@property` read alias on `UserPreferences`.
- Added fields: `avoided_interests: List[str]` (persisted in `avoided_interests_json`) and `travel_habits: List[str]`.

### Deprecated Wrappers
- `ProfileStore` and `MemoryLoader` are deprecated thin wrappers delegating to `ProfileService.default()`.
- `ProfileStore.ensure_user_profile` raises `UnknownUserError` if the user is not registered in `users`.
- `MOCK_USER_DB` is completely removed.

### Nationality Invariant
- `nationality` is `Optional[str] = None` end-to-end.
- `None` means unknown nationality; downstream agents must ask the user rather than assuming Egyptian.
