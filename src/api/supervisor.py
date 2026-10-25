# app/api/supervisor.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.runner import execute_graph_sync

router = APIRouter(prefix="/supervisor", tags=["Supervisor Agent"])

# Pydantic schemas for strict payload validation
class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    routed_to: str
    current_location: str | None = None

@router.post("/chat", response_model=ChatResponse)
def supervisor_chat(request: ChatRequest):
    """
    Main entry point for user inquiries. 
    Synchronously routes the request through the Supervisor LangGraph.
    """
    # 1. Construct the initial state schema (Dev 3)
    initial_state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "messages": [("user", request.message)],
        "current_intent": "unclassified"
    }
    
    # 2. Execute the graph ( Dev 4 )
    # from app.agents.supervisor_graph import supervisor_app
    # final_state = execute_graph_sync(supervisor_app, initial_state)
    
    # 3. Temporary mock response for testing
    final_state = {
        "response": f"Acknowledged: '{request.message}'. Graph execution pending.",
        "routed_to": "profiling_stub"
    }
    
    return ChatResponse(
        response=final_state["response"],
        routed_to=final_state["routed_to"]
    )