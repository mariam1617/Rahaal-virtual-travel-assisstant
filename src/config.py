# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_name: str = "Travel Assistant Backend"
    version: str = "0.1.0"
    
    # required API keys or DB URIs here later
    
    # This tells Pydantic to look for a .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()