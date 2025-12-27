"""
Evaluation Module - System Performance Lab.

Demonstrates:
- RAG Pipeline Evaluation
- Latency Tracking
- Quality Metrics (Faithfulness, Relevance)
"""
import streamlit as st
import time
import pandas as pd
from core.evals.engine import RAGEvaluator

def render():
    st.markdown("## 🧪 Evaluation Lab")
    st.markdown("Test and verify the performance of your AI pipelines.")

    tab1, tab2 = st.tabs(["⚡ Quick Eval", "📊 Golden Dataset Run"])

    with tab1:
        st.subheader("Single Query Analysis")
        
        # Inputs
        query = st.text_input("Test Question", "What represents 40% of the project's value?")
        context = st.text_area("Retrieved Context (Paste or Auto-fill)", 
                             "RAG implementation is 40% of the project value. It includes vector DBs and embedding pipelines.")
        generated_answer = st.text_area("Generated Answer", 
                                      "RAG implementation accounts for 40% of the project value.")
        ground_truth = st.text_input("Ground Truth (Optional)", "RAG systems are 40%.")

        if st.button("Run Evaluation"):
            import requests
            import os
            
            with st.spinner("Judging response via API..."):
                try:
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    payload = {
                        "query": query,
                        "answer": generated_answer,
                        "context": context,
                        "ground_truth": ground_truth if ground_truth else None
                    }
                    
                    response = requests.post(f"{backend_url}/eval/evaluate", json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        scores = data
                        latency = data.get("latency", 0)
                        
                        # Display Results
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                        if ground_truth:
                            c2.metric("Correctness", f"{scores.get('correctness', 0):.2f}")
                        c3.metric("Latency", f"{latency:.3f}s")
                        
                        st.info("Evaluation performed by **AgentForge API**.")
                    else:
                        st.error(f"API Error: {response.text}")
                
                except Exception as e:
                    st.warning(f"Backend unreachable ({e}). Falling back to local execution...")
                    evaluator = RAGEvaluator(provider="gemini")
                    start_time = time.time()
                    scores = evaluator.evaluate_response(query, generated_answer, context, ground_truth)
                    latency = time.time() - start_time
                    
                    # Display Results
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                    if ground_truth:
                        c2.metric("Correctness", f"{scores.get('correctness', 0):.2f}")
                    c3.metric("Latency", f"{latency:.3f}s")

    with tab2:
        st.subheader("Batch Evaluation")
        st.info("Coming soon: Run full test suite against `tests/data/golden_dataset.json`")
