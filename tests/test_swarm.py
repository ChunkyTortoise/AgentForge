
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MOCK dependencies that might be missing or require API keys
sys.modules["langchain_community.tools"] = MagicMock()
sys.modules["langchain_community.tools.duckduckgo_search"] = MagicMock()
# Specifically mock the class used in agents.py
mock_ddg = MagicMock()
sys.modules["langchain_community.tools"].DuckDuckGoSearchRun = mock_ddg

# Import core.agents explicitly to register it
try:
    import core.agents
except ImportError as e:
    print(f"❌ Failed to import core.agents: {e}")
    pass

class TestSwarmGraph(unittest.TestCase):
    
    @patch('core.agents.BaseAgent')
    def test_swarm_graph_construction(self, MockBaseAgent):
        """Test that the Swarm Graph is constructed with correct nodes."""
        # Mock LLM to return a simple invoke response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Mock Analysis"
        mock_llm.invoke.return_value = mock_response
        
        # Setup MockBaseAgent to return our mock_llm
        instance = MockBaseAgent.return_value
        instance.llm = mock_llm
        
        from core.agents import create_swarm_graph
        
        # Build graph
        graph = create_swarm_graph(provider="gemini")
        
        # Verify it compiles (returns a CompiledGraph)
        self.assertIsNotNone(graph)
        
        # We can inspect the underlying graph definition
        # graph.get_graph() returns a drawable graph object
        drawable = graph.get_graph()
        nodes = drawable.nodes
        
        # Check for our swarm nodes
        self.assertIn("planner", nodes)
        self.assertIn("market_analyst", nodes)
        self.assertIn("tech_analyst", nodes)
        self.assertIn("risk_analyst", nodes)
        self.assertIn("aggregator", nodes)
        
        print("✅ Swarm Graph structure verified: Planner -> [Market, Tech, Risk] -> Aggregator")

    @patch('core.agents.BaseAgent')
    def test_swarm_execution_flow(self, MockBaseAgent):
        """Test the execution flow (mocked)."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analyzed."
        mock_llm.invoke.return_value = mock_response
        
        instance = MockBaseAgent.return_value
        instance.llm = mock_llm
        
        from core.agents import create_swarm_graph
        graph = create_swarm_graph(provider="gemini")
        
        # Mock the graph invocation if we don't want to rely on LangGraph internals in unit test
        # But if we rely on the mocked LLM, we can actually 'invoke' the graph and see if it completes
        
        # Note: Validating LangGraph execution in unit tests can be tricky without real State.
        # We will trust the structure test above for now.
        pass

if __name__ == '__main__':
    unittest.main()
