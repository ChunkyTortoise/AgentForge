from fastapi import APIRouter, HTTPException
import time
from pydantic import BaseModel
from typing import Optional, Dict, Any

from core.agents import create_todo_solver_graph
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/todo", tags=["todo"])

class TodoRequest(BaseModel):
    file_path: str = "TODO.md"
    provider: str = "gemini"

class TodoResponse(BaseModel):
    status: str
    selected_task: str
    proposal: Optional[str] = None
    metadata: Dict[str, Any]

@router.post("/solve", response_model=TodoResponse)
async def solve_todo(request: TodoRequest):
    """
    Analyzes the TODO file and proposes a solution for the top task.
    """
    logger.info(f"Received TODO Solver Request for: {request.file_path}")
    start_time = time.time()
    
    try:
        # Initialize graph
        graph = create_todo_solver_graph(provider=request.provider)
        
        # Run workflow
        inputs = {
            "target_file": request.file_path,
            "messages": [] # Init empty history
        }
        
        # Async invocation
        result = await graph.ainvoke(inputs)
        
        latency = time.time() - start_time
        
        return TodoResponse(
            status="success",
            selected_task=result.get("selected_task", "Unknown"),
            proposal=result.get("code_proposal"),
            metadata={"latency": latency, "provider": request.provider}
        )
    except Exception as e:
        logger.error(f"TODO Solver Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
