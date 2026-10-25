# app/api/supervisor.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.runner import execute_graph_sync
from src.agents.supervisor.router import create_supervisor_graph

router = APIRouter(prefix="/supervisor", tags=["Supervisor Agent"])

# Compile the supervisor graph once at module load
supervisor_app = create_supervisor_graph()

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
    # 1. Construct the initial state
    initial_state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "messages": [("user", request.message)],
        "current_intent": "unclassified"
    }
    
    # 2. Execute the graph (Dev 4)
    final_state = execute_graph_sync(supervisor_app, initial_state)
    
    response_text = final_state.get(
        "response",
        f"Request routed to {final_state.get('next_agent', 'specialist')}."
    )
    routed_to = final_state.get("next_agent", "unknown")
    
    return ChatResponse(
        response=response_text,
        routed_to=routed_to,
        current_location=final_state.get("current_location")
    )