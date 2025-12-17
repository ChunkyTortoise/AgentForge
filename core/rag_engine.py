"""
RAG Engine - Vector Database and Retrieval Logic.

Demonstrates:
- ChromaDB integration
- Semantic search implementation
- Document management
"""
import os
import shutil
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import uuid

import chromadb
from chromadb.config import Settings

from core.embeddings import EmbeddingModel
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Standardized search result."""
    text: str
    source: str
    id: str
    distance: float
    metadata: Dict[str, Any]


class VectorStore:
    """
    Wrapper around ChromaDB for document storage and retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "agentforge_docs",
        embedding_model: Optional[EmbeddingModel] = None,
        persist_directory: str = "./.chroma_db"
    ):
        """
        Initialize the Vector Store.
        
        Args:
            collection_name: Name of the Chroma collection
            embedding_model: Instance of EmbeddingModel class
            persist_directory: Path to save the database
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Initialize Client
        self._client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or Create Collection
        # Note: Chroma handles the embedding function adapter, but we'll manage embeddings explicitly 
        # to simplify swapping providers in our clean architecture.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"VectorStore initialized at {persist_directory}, Collection: {collection_name}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        """
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        if not metadatas:
            metadatas = [{"source": "unknown"} for _ in texts]
            
        try:
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Add to Chroma
            self._collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        mode: str = "semantic" # semantic, keyword, hybrid
    ) -> List[SearchResult]:
        """
        Search for relevant documents.
        Supports 'semantic' (default), 'keyword' (exact match boost), or 'hybrid'.
        """
        try:
            # 1. Semantic Search (Base)
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2 if mode == "hybrid" else n_results,
                where=filter_metadata
            )
            
            # Parse Vector Results
            candidates = {}
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    candidates[doc_id] = SearchResult(
                        id=doc_id,
                        text=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        source=results['metadatas'][0][i].get('source', 'unknown') if results['metadatas'] else 'unknown',
                        distance=results['distances'][0][i] if results['distances'] else 0.0
                    )

            if mode in ["keyword", "hybrid"]:
                # 2. Simple Keyword Scoring (Reranking)
                # In a real system, we'd query an Inverted Index. here we scan candidates.
                # Just rerank the semantic candidates based on keyword overlap
                query_terms = set(query.lower().split())
                
                for doc_id, res in candidates.items():
                    doc_lower = res.text.lower()
                    score = sum(1 for term in query_terms if term in doc_lower)
                    # Normalize simple score
                    keyword_boost = score * 0.1 
                    # Reduce distance (which is bad in cosine distance, usually smaller is better? 
                    # Chroma returns distance. If cosine DISTANCE, 0 is identical.
                    # We want to reduce distance if keywords match.
                    res.distance = max(0.0, res.distance - keyword_boost)

            # Sort and Slice
            final_results = list(candidates.values())
            final_results.sort(key=lambda x: x.distance)
            
            return final_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
