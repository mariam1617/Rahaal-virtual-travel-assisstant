# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting up {settings.project_name}...")
    # TODO (Dev 1/3): Initialize synchronous DB connection pools or LangGraph state models here
    
    yield  # Application is now running and accepting requests
    
    # --- Shutdown Logic ---
    print("Shutting down cleanly...")
    # TODO: Close connections and clean up resources

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    lifespan=lifespan
)

