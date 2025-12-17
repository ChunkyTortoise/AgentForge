
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock streamlit to avoid runtime errors during import
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.runtime.scriptrunner"] = MagicMock()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModuleImports(unittest.TestCase):
    def test_core_imports(self):
        """Test that core modules import without error."""
        try:
            from core.agents import BaseAgent
            from core.embeddings import EmbeddingModel
            from core.rag_engine import VectorStore
            from core.llm_client import LLMClient
            print("✅ Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")

    def test_module_imports(self):
        """Test that feature modules import without error."""
        try:
            from modules import smart_analyst
            from modules import rag_assistant
            # from modules import agent_hub # Still partial
            print("✅ Feature modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import feature modules: {e}")

    def test_rag_engine_init(self):
        """Test initialization of RAG engine (mocked)."""
        # This tests if the class structure is valid, not actual DB logic
        try:
            from core.rag_engine import VectorStore
            # We don't instantiate to avoid creating DB files
            self.assertTrue(hasattr(VectorStore, 'add_texts'))
            print("✅ RAG Engine class structure verified")
        except Exception as e:
            self.fail(f"RAG Engine validation failed: {e}")

if __name__ == '__main__':
    unittest.main()
