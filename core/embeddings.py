"""
Embeddings Interface - Support for Local and API-based Embeddings.

Demonstrates:
- Vector Database fundamentals
- Hybrid embedding strategies (Cost vs. Performance)
"""
import os
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    # LOCAL = "local"  # Disabled due to Python 3.13 torch incompatibility
    GOOGLE = "google"  # Google Generative AI


class EmbeddingModel:
    """
    Unified interface for embedding generation.
    
    Defaults to Google embeddings (API) as local support is currently limited.
    """
    
    def __init__(self, provider: str = "google", model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            provider: 'google' (default)
            model_name: Specific model string
        """
        self.provider = EmbeddingProvider(provider)
        self.model_name = model_name
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the specific embedding client."""
        try:
            if self.provider == EmbeddingProvider.GOOGLE:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found. Please set it in .env")
                
                model = self.model_name or "models/embedding-001"
                self._client = GoogleGenerativeAIEmbeddings(
                    model=model,
                    google_api_key=api_key
                )
                logger.info(f"Initialized Google Embeddings: {model}")
            else:
                 logger.warning("Local embeddings disabled. Please use 'google'.")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of vector embeddings
        """
        if not self._client:
            raise RuntimeError("Embedding client not initialized")
            
        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise RuntimeError(f"Failed to embed documents: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: Query string
            
        Returns:
            Vector embedding
        """
        if not self._client:
            raise RuntimeError("Embedding client not initialized")
            
        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise RuntimeError(f"Failed to embed query: {e}")
