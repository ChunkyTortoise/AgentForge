"""
Simple integration test for AgentForge API.
"""
from fastapi.testclient import TestClient
from api.main import app
import unittest
from unittest.mock import patch, MagicMock

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "operational")

    @patch("api.main.create_swarm_graph")
    def test_swarm_endpoint(self, mock_create_graph):
        # Mock graph invocation
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_report": "Test Report",
            "market_analysis": "Market OK",
            "technical_feasibility": "Tech OK",
            "risk_assessment": "Risk Low"
        }
        mock_create_graph.return_value = mock_graph
        
        response = self.client.post("/swarm/run", json={"task": "test", "provider": "gemini"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["final_report"], "Test Report")

if __name__ == "__main__":
    unittest.main()
