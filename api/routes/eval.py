from fastapi import APIRouter, HTTPException
import time
from api.schemas import EvalRequest, EvalResponse
from core.evals.engine import RAGEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_rag(request: EvalRequest):
    """
    Evaluate a RAG response using the Eval Engine.
    """
    start_time = time.time()
    evaluator = RAGEvaluator()
    
    try:
        # evaluator.evaluate_response is now asynchronous.
        scores = await evaluator.evaluate_response(
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
        logger.error(f"Evaluation Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
