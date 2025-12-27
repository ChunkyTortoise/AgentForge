"""
Integration test for Agentic TODO Solver API.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from api.main import app

class TestTodoAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("api.routes.todo.create_todo_solver_graph")
    def test_solve_endpoint(self, mock_create_graph):
        # Mock graph invocation with AsyncMock for ainvoke
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "selected_task": "Fix Bug X",
            "code_proposal": "## Plan\n1. Edit file\n## Code Changes\n..."
        })
        mock_create_graph.return_value = mock_graph
        
        response = self.client.post("/todo/solve", json={"file_path": "TODO.md", "provider": "gemini"})
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selected_task"], "Fix Bug X")
        self.assertIn("## Plan", response.json()["proposal"])
        
        # Verify ainvoke was called
        mock_graph.ainvoke.assert_called_once()

if __name__ == "__main__":
    unittest.main()

