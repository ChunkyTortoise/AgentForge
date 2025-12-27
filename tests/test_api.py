"""
Integration test for AgentForge API (Async compatible).
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("operational", response.json()["status"])

    @patch("api.routes.swarm.create_swarm_graph")
    def test_swarm_endpoint(self, mock_create_graph):
        # Mock graph invocation with AsyncMock for ainvoke
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_report": "Test Report",
            "market_analysis": "Market OK",
            "technical_feasibility": "Tech OK",
            "risk_assessment": "Risk Low"
        })
        mock_create_graph.return_value = mock_graph
        
        response = self.client.post("/swarm/run", json={"task": "test", "provider": "gemini"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["final_report"], "Test Report")
        # Verify ainvoke was called
        mock_graph.ainvoke.assert_called_once()

    @patch("api.routes.eval.RAGEvaluator")
    def test_eval_endpoint(self, mock_evaluator_cls):
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_response = AsyncMock(return_value={
            "faithfulness": 1.0,
            "correctness": 0.9
        })
        mock_evaluator_cls.return_value = mock_evaluator
        
        payload = {
            "query": "test",
            "answer": "test",
            "context": "test"
        }
        response = self.client.post("/eval/evaluate", json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["faithfulness"], 1.0)

if __name__ == "__main__":
    unittest.main()