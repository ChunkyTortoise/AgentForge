import time
from fastapi import FastAPI, Request
from api.routes import swarm, eval, todo, chat
from core.config import settings
from utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Backend API for AgentForge AI Platform",
    version=settings.version
)

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """
    Middleware for Correlation ID tracking and performance monitoring.
    """
    # 1. Track Correlation ID
    incoming_cid = request.headers.get("X-Correlation-ID")
    cid = set_correlation_id(incoming_cid)
    
    # 2. Track Time
    start_time = time.time()
    
    # 3. Process Request
    response = await call_next(request)
    
    # 4. Finalize Metrics
    duration = time.time() - start_time
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    response.headers["X-Correlation-ID"] = cid
    
    logger.info(f"DONE | {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration:.4f}s")
    return response

@app.get("/")
async def root():
    return {
        "message": f"{settings.app_name} API is online", 
        "version": settings.version,
        "status": "operational"
    }

# Include Routers
app.include_router(swarm.router)
app.include_router(eval.router)
app.include_router(todo.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
