"""
Evaluation Engine for RAG Systems.

Demonstrates:
- Golden Dataset comparison
- Semantic Similarity scoring (using Cosine Similarity of embeddings)
- Faithfulness checks (LLM-as-a-Judge)
"""
import numpy as np
from typing import Dict
from core.llm_client import LLMClient
from core.embeddings import EmbeddingModel

class RAGEvaluator:
    def __init__(self, provider=None):
        self.llm_client = LLMClient(provider=provider)
        self.embedding_model = EmbeddingModel()
        
    async def evaluate_response(self, question: str, answer: str, context: str, ground_truth: str = None) -> Dict[str, float]:
        """
        Evaluate a single RAG response (Asynchronous).
        Returns dictionary of scores (0.0 to 1.0).
        """
        metrics = {}
        
        # 1. Answer Relevance (Cosine Similarity with Ground Truth)
        if ground_truth:
            metrics["correctness"] = self._calculate_similarity(answer, ground_truth)
        
        # 2. Faithfulness (LLM as Judge) - Async call
        metrics["faithfulness"] = await self._check_faithfulness(answer, context)
        
        return metrics

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        vec1 = self.embedding_model.embed_query(text1)
        vec2 = self.embedding_model.embed_query(text2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))

    async def _check_faithfulness(self, answer: str, context: str) -> float:
        """
        Uses an LLM to judge if the answer is faithful to the context.
        Returns 1.0 (Faithful) or 0.0 (Hallucination).
        """
        prompt = f"""
        You are a strict Fact-Checking Judge.
        
        CONTEXT:
        {context[:4000]}
        
        ANSWER:
        {answer}
        
        TASK:
        Determine if the ANSWER is entirely supported by the CONTEXT.
        If yes, return "1.0". If it contains information NOT in context, return "0.0".
        Return ONLY the number.
        """
        try:
            response = await self.llm_client.agenerate(prompt, temperature=0.0)
            score = float(response.content.strip())
            return score
        except Exception:
            return 0.5 # Default to unsure