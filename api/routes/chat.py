from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from core.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"
    system_prompt: Optional[str] = None

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response from the LLM.
    Returns a Server-Sent Events (SSE) compatible stream (or just raw text chunks).
    """
    logger.info(f"Received Chat Stream Request: {request.message[:50]}...")
    
    try:
        client = LLMClient(provider=request.provider)
        
        async def event_generator():
            try:
                async for chunk in client.astream(request.message, system_prompt=request.system_prompt):
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"[ERROR: {str(e)}]"

        return StreamingResponse(event_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Chat Init Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
