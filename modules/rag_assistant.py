"""
RAG Assistant Module - Strategic Document Research.

Demonstrates:
- IBM RAG & Agentic AI (Advanced Retrieval)
- Duke LLMOps (Vector Databases)
- Multi-step reasoning over documents
"""
import os
import shutil
import tempfile
import streamlit as st

from langchain_core.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Core imports
from core.rag_engine import VectorStore
from core.agents import BaseAgent
from core.llm_client import get_available_providers
from utils.logger import get_logger

logger = get_logger(__name__)

def render() -> None:
    """Render the RAG Assistant module."""
    st.markdown("""
    **Strategic Research Assistant**: An agentic RAG system that executes **multi-step reasoning** 
    to answer complex questions from your documents.
    
    **Skills**: IBM RAG/Agentic AI, Vector Databases (ChromaDB), LangGraph
    """)

    # 1. Setup
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    
    # Check Providers
    providers = get_available_providers()
    if not any(providers.values()):
        st.warning("⚠️ No LLM API keys found.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Knowledge Base")
        
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🔄 Process & Index Documents"):
                with st.spinner(" ingest ing..."):
                    vs = _ingest_documents(uploaded_files)
                    if vs:
                        st.session_state.rag_vector_store = vs
                        
                        # Initialize Agent with Retrieval Tool
                        tools = [
                            Tool(
                                name="search_knowledge_base",
                                func=lambda q: args_wrapper(vs, q),
                                description="Always use this to search the uploaded documents for specific information."
                            )
                        ]
                        
                        # Select Provider
                        provider = "claude" if providers.get("claude") else "gemini"
                        st.session_state.rag_agent = BaseAgent(tools=tools, provider=provider)
                        st.success(f"✅ Indexed {len(uploaded_files)} files.")

        if "rag_vector_store" in st.session_state:
            st.info(f"📚 Knowledge Base Active")
            if st.button("Clear Knowledge Base"):
                del st.session_state.rag_vector_store
                st.session_state.rag_agent = None
                st.experimental_rerun()

    with col2:
        st.subheader("2. Strategic Inference")
        
        # Chat Interface
        for msg in st.session_state.rag_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        query = st.chat_input("Ask a complex question (e.g., 'Compare the risk factors in these documents vs industry norms')")
        
        if query:
            if not st.session_state.rag_agent:
                st.error("Please upload and index documents first.")
                return

            # User Msg
            st.session_state.rag_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Agent Msg
            with st.chat_message("assistant"):
                status_container = st.empty()
                status_container.info("🤔 Converting to Agentic Graph...")
                
                try:
                    # 1. Imports
                    from core.rag_agent import create_rag_graph
                    
                    # 2. Define Retriever Wrapper
                    # The graph expects a callable that takes a query -> returns docs
                    def retriever_func(q):
                        return st.session_state.rag_vector_store.search(q, n_results=5)
                    
                    provider = "gemini" if "gemini" in providers.keys() else "claude"
                    app = create_rag_graph(db_retriever=retriever_func, provider=provider)
                    
                    # 3. Execute
                    inputs = {"question": query, "retry_count": 0}
                    
                    # Streaming (Simulator)
                    # For a real app, we'd use app.stream() iterator
                    final_generation = ""
                    
                    status_container.info("🔄 Agent Steps: Retrieving -> Grading relevance...")
                    
                    # Run full invocation
                    for output in app.stream(inputs):
                        for key, value in output.items():
                            # Show status updates based on active node
                            if key == "retrieve":
                                status_container.info("🔍 Retrieved documents from VectorDB.")
                            elif key == "grade_documents":
                                num_relevant = len(value.get("documents", []))
                                if num_relevant == 0:
                                    status_container.warning("❌ No relevant docs found. Self-correcting...")
                                else:
                                    status_container.success(f"✅ Found {num_relevant} relevant documents.")
                            elif key == "transform_query":
                                new_q = value.get("question")
                                status_container.warning(f"🔄 Rewriting query to: '{new_q}'")
                            elif key == "generate":
                                final_generation = value.get("generation")
                                
                    # 4. Display Final
                    if final_generation:
                        status_container.empty() # clear status
                        message_content = f"**Answer:**\n\n{final_generation}"
                        st.markdown(message_content)
                        st.session_state.rag_chat_history.append({"role": "assistant", "content": message_content})
                    else:
                        status_container.error("Failed to generate an answer.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"RAG Error: {e}")

def args_wrapper(vs, query):
    """Helper to format search results as string."""
    results = vs.search(query, n_results=4)
    return "\n\n".join([f"[Source: {r.source}]\n{r.text}" for r in results])

def _ingest_documents(uploaded_files) -> VectorStore:
    """Ingest uploaded files into ChromaDB."""
    try:
        # Save to temp
        temp_dir = tempfile.mkdtemp()
        docs_to_index = []
        
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load & Split
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(raw_docs)
            
            # Extract text and metadata
            for chunk in chunks:
                docs_to_index.append({
                    "text": chunk.page_content,
                    "metadata": {"source": uploaded_file.name, "page": chunk.metadata.get("page", 0)}
                })
        
        # Index
        vs = VectorStore(
            collection_name=f"rag_collection_{len(uploaded_files)}",
            persist_directory="./.chroma_rag"
        )
        
        texts = [d["text"] for d in docs_to_index]
        metadatas = [d["metadata"] for d in docs_to_index]
        
        vs.add_texts(texts, metadatas=metadatas)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return vs

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        st.error(f"Failed to process documents: {e}")
        return None
