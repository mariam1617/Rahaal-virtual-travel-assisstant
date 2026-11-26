
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "Travel Assistant Backend"
    version: str = "0.1.0"

    # required API keys or DB URIs here later
    openmap_api_key: Optional[str] = None  # Set in .env when using a paid OpenMap/Nominatim service
    geocoder_provider: str = "nominatim"
    geocoder_base_url: str = "https://nominatim.openstreetmap.org/reverse"
    geocoder_contact: str = ""
    geocoder_timeout_s: float = 5.0
    geocoder_cache_ttl_s: int = 86400
    geocoder_fail_ttl_s: int = 30
    geocoder_min_interval_s: float = 1.0

    # This tells Pydantic to look for a .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()