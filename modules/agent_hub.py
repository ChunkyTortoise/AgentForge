"""
Agent Hub Module - Multi-Agent Research System.

Demonstrates:
- IBM RAG and Agentic AI
- Google Cloud GenAI Agents
- LangGraph and CrewAI concepts
"""
import streamlit as st
import os
import requests
from typing import Optional

from core.llm_client import LLMClient
from utils.logger import get_logger
from langchain_core.messages import HumanMessage, AIMessage

logger = get_logger(__name__)


def render() -> None:
    """Render the Agent Hub module."""
    st.markdown("""
    # Workflow Selection
    """)
    st.markdown("### 🛠️ Agent Workflows")
    
    workflow_type = st.selectbox(
        "Select Workflow",
        ["Research & Report", "Parallel Swarm Strategy", "Agentic TODO Solver"]
    )
    
    if workflow_type == "Research & Report":
        _render_research_workflow()
    elif workflow_type == "Parallel Swarm Strategy":
        _render_swarm_workflow()
    elif workflow_type == "Agentic TODO Solver":
        _render_todo_solver()

def _render_todo_solver() -> None:
    """Render the Agentic TODO Solver workflow."""
    st.markdown("#### ✅ Agentic TODO Solver")
    st.markdown("An autonomous agent that reads `TODO.md`, investigates the codebase, and proposes code changes.")
    
    file_path = st.text_input("Target File", "TODO.md")
    
    if st.button("🚀 Solve Next Task"):
        with st.spinner("Agent is scanning codebase and formulating a plan..."):
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                payload = {
                    "file_path": file_path,
                    "provider": "gemini"
                }
                
                response = requests.post(f"{backend_url}/todo/solve", json=payload, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"Analyzed Task: {data.get('selected_task')}")
                    
                    with st.expander("📄 View Proposed Solution", expanded=True):
                        st.markdown(data.get("proposal", "No proposal generated."))
                        
                    st.info(f"Latency: {data.get('metadata', {}).get('latency', 0):.2f}s")
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")

def _render_swarm_workflow() -> None:
    """Render the Strategic Swarm workflow (Calling FastAPI Backend)."""
    st.markdown("#### 🐝 Strategic Swarm (Multi-Agent Parallel)")
    st.markdown("This workflow triggers a **Planner** who fans out to **Market**, **Tech**, and **Risk** analysts in parallel.")
    
    topic = st.text_input("Enter Topic for Analysis", "The future of Agentic AI in healthcare")
    
    if st.button("🚀 Launch Swarm"):
        with st.spinner("Swarm agents are collaborating via FastAPI backend..."):
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                
                payload = {
                    "task": topic,
                    "provider": "gemini"
                }
                
                response = requests.post(f"{backend_url}/swarm/run", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Swarm Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(data.get("final_report", "No report generated."))
                    with col2:
                        with st.expander("🔍 View Raw Analyst Outputs"):
                            st.write("**Market Analysis**")
                            st.info(data.get("market_analysis", "N/A"))
                            st.write("**Technical Analysis**")
                            st.info(data.get("technical_feasibility", "N/A"))
                            st.write("**Risk Assessment**")
                            st.info(data.get("risk_assessment", "N/A"))
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")

def _render_research_workflow() -> None:
    """Render the Research & Report workflow interface."""
    st.markdown("#### 🕵️ Research Team (Researcher + Writer)")
    st.markdown("A multi-agent system where a **Researcher** gathers facts and a **Writer** creates a blog post.")
    
    topic = st.text_input("Research Topic", placeholder="e.g., The future of Generative AI in Healthcare")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("LLM Provider", ["gemini", "claude"], key="agent_provider")
    with col2:
        st.info("Workflow: [Start] -> Researcher (Tools) -> Writer -> [End]")
        
    if st.button("🚀 Start Research Team"):
        if not topic:
            st.warning("Please enter a research topic.")
            return

        from core.agents import create_research_graph
        try:
            graph = create_research_graph(provider=provider)
        except Exception as e:
            st.error(f"Failed to initialize graph: {e}")
            return
            
        col_status, col_graph = st.columns([1, 1])
        with col_graph:
            with st.expander("View Graph Topology", expanded=True):
                try:
                    graph_img = graph.get_graph().draw_mermaid_png()
                    st.image(graph_img, caption="Multi-Agent Workflow", use_container_width=True)
                except Exception:
                    st.warning("Could not generate graph image.")

        with col_status:
            if "research_thread_id" not in st.session_state:
                st.session_state.research_thread_id = "thread_1"
            
            thread_config = {"configurable": {"thread_id": st.session_state.research_thread_id}}
            
            st.info(f"🔄 **Team Active**: Researcing '{topic}'...")
            status_container = st.empty()
            
            if st.button("▶️ Run / Resume Workflow"):
                status_container.text("Agents working...")
                try:
                    current_state = graph.get_state(thread_config)
                    if not current_state.values:
                         graph.invoke({"messages": [("user", f"Research this topic: {topic}")]}, thread_config)
                    else:
                        graph.invoke(None, thread_config)
                except Exception as e:
                    pass
            
            state_snapshot = graph.get_state(thread_config)
            
            if state_snapshot.next:
                if "manager" in state_snapshot.next:
                    st.warning("⚠️ **Manager Review Required**")
                    last_msg = state_snapshot.values["messages"][-1]
                    st.markdown("### 🔎 Researcher Findings")
                    st.markdown(last_msg.content)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Approve"):
                            graph.update_state(thread_config, {"messages": [HumanMessage(content="APPROVE")]}, as_node="manager")
                            st.rerun() 
                            st.success("Approved! Click 'Run/Resume' to proceed.")
                    with c2:
                        feedback = st.text_input("Feedback for Rejection")
                        if st.button("↩️ Reject"):
                             graph.update_state(thread_config, {"messages": [HumanMessage(content=f"REJECT: {feedback}")]}, as_node="manager")
                             st.error("Rejected sent. Click 'Run/Resume' to retry.")

            if not state_snapshot.next and state_snapshot.values:
                 final_msg = state_snapshot.values["messages"][-1]
                 status_container.success("✅ Work Complete!")
                 st.markdown("### 📄 Final Report")
                 st.markdown(final_msg.content)