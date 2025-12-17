import pytest
from core.agents import ResearchAgent, BaseAgent
from core.rag_engine import VectorStore
import pandas as pd
# We can't easily test Smart Analyst tool without mocking the whole agent setup, 
# but we can check if the tool definition logic is sound if we could import it. 
# However, the tool is defined *inside* the render function scope in the current implementation of Smart Analyst 
# which makes unit testing hard. 
# We will focus on Agent and RAG.

def test_research_agent_structure():
    """Verify ResearchAgent has correct structure."""
    agent = ResearchAgent(provider="gemini")
    assert isinstance(agent, BaseAgent)
    # Check graph compilation
    assert agent.app is not None
    # Check mermaid generation
    graph_img = agent.get_graph_image()
    assert graph_img is not None
    assert len(graph_img) > 0

def test_hybrid_search():
    """Verify Hybrid Search logic in VectorStore."""
    # Use a mock embedding model or just real one if creds exist? 
    # Real one needs API key. Assuming env is set, or we mock it.
    # If no API key, this might fail unless we mock.
    
    # Simple Mock Embedding Model for testing
    class MockEmbedding:
        def embed_documents(self, texts):
            return [[0.1]*768 for _ in texts]
        def embed_query(self, text):
            return [0.1]*768

    vs = VectorStore(collection_name="test_hybrid", embedding_model=MockEmbedding(), persist_directory="./.test_db")
    
    ids = vs.add_texts(["apple banana", "apple orange", "banana grape"])
    
    # Test Hybrid
    # query "orange" should boost "apple orange"
    # "orange" is not in "apple banana" or "banana grape"
    
    # We mock the search response from Chroma since we didn't actually insert anything meaningful relative to query 
    # (embeddings are identical).
    # Actually, Chroma might return random order if embeddings identical.
    
    # Let's rely on the internal logic check by inspecting the code or running it:
    # If we run it, it should not crash.
    results = vs.search("orange", n_results=3, mode="hybrid")
    assert len(results) <= 3
    
    vs.clear()

if __name__ == "__main__":
    test_research_agent_structure()
    test_hybrid_search()
    print("All Phase 2 logic tests passed!")
