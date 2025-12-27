from fastapi import APIRouter, HTTPException
import time
from typing import Optional, Dict, Any
from api.schemas import SwarmRequest, SwarmResponse
from core.agents import create_swarm_graph
from api.tasks import run_swarm_task
from celery.result import AsyncResult
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/swarm", tags=["swarm"])

@router.post("/run", response_model=SwarmResponse)
async def run_swarm_sync(request: SwarmRequest):
    """
    Execute a Strategic Swarm workflow synchronously (blocking).
    """
    logger.info(f"Received Sync Swarm Request: {request.task}")
    start_time = time.time()
    try:
        graph = create_swarm_graph(provider=request.provider)
        result = await graph.ainvoke({"topic": request.task})
        latency = time.time() - start_time
        return SwarmResponse(
            status="success",
            plan=result.get("plan"),
            market_analysis=result.get("market_analysis"),
            technical_feasibility=result.get("technical_feasibility"),
            risk_assessment=result.get("risk_assessment"),
            final_report=result.get("final_report"),
            metadata={"latency": latency, "provider": request.provider}
        )
    except Exception as e:
        logger.error(f"Sync Swarm Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit")
async def submit_swarm(request: SwarmRequest):
    """
    Submit a swarm workflow to the distributed task queue (non-blocking).
    Returns a job_id for polling.
    """
    logger.info(f"Submitting Swarm Task to Queue: {request.task}")
    task = run_swarm_task.delay(request.task, request.provider)
    return {"job_id": task.id, "status": "queued"}

@router.get("/status/{job_id}")
async def get_swarm_status(job_id: str):
    """
    Check the status and result of a queued swarm task.
    """
    task_result = AsyncResult(job_id)
    
    response = {
        "job_id": job_id,
        "status": task_result.status,
    }
    
    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        else:
            response["error"] = str(task_result.result)
            
    return response