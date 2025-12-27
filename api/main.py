import time
from fastapi import FastAPI, HTTPException, Request
import time
from api.schemas import SwarmRequest, SwarmResponse, EvalRequest, EvalResponse
from core.agents import create_swarm_graph
from core.evals.engine import RAGEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AgentForge API",
    description="Backend API for AgentForge AI Platform",
    version="1.0.0"
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Path: {request.url.path} | Duration: {process_time:.4f}s")
    return response

@app.get("/")
async def root():
    return {"message": "AgentForge API is online", "status": "operational"}

@app.post("/swarm/run", response_model=SwarmResponse)
async def run_swarm(request: SwarmRequest):
    """
    Execute a Strategic Swarm workflow via the LangGraph engine.
    """
    logger.info(f"Received Swarm Request: {request.task}")
    start_time = time.time()
    
    try:
        # Initialize graph
        graph = create_swarm_graph(provider=request.provider)
        
        # Run workflow
        inputs = {"task": request.task}
        result = graph.invoke(inputs)
        
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
        logger.error(f"Swarm Execution Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval/evaluate", response_model=EvalResponse)
async def evaluate_rag(request: EvalRequest):
    """
    Evaluate a RAG response using the Eval Engine.
    """
    start_time = time.time()
    evaluator = RAGEvaluator()
    
    try:
        scores = evaluator.evaluate_response(
            question=request.query,
            answer=request.answer,
            context=request.context,
            ground_truth=request.ground_truth
        )
        
        latency = time.time() - start_time
        
        return EvalResponse(
            faithfulness=scores.get("faithfulness", 0.0),
            correctness=scores.get("correctness"),
            latency=latency
        )
    except Exception as e:
        logger.error(f"Evaluation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
