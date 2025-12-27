from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SwarmRequest(BaseModel):
    task: str
    provider: str = "gemini"

class SwarmResponse(BaseModel):
    status: str
    plan: Optional[str] = None
    market_analysis: Optional[str] = None
    technical_feasibility: Optional[str] = None
    risk_assessment: Optional[str] = None
    final_report: str
    metadata: Dict[str, Any]

class EvalRequest(BaseModel):
    query: str
    answer: str
    context: str
    ground_truth: Optional[str] = None

class EvalResponse(BaseModel):
    faithfulness: float
    correctness: Optional[float] = None
    latency: float
