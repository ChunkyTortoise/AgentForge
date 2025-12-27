"""
Agent Hub Module - Multi-Agent Research System.

Demonstrates:
- IBM RAG and Agentic AI
- Google Cloud GenAI Agents
- LangGraph and CrewAI concepts
"""
import streamlit as st
import os
from typing import Optional

from core.llm_client import LLMClient
from utils.logger import get_logger

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
        ["Research & Report", "Parallel Swarm Strategy", "Codebase Analyst"]
    )
    
    if workflow_type == "Research & Report":
        _render_research_workflow()
    elif workflow_type == "Parallel Swarm Strategy":
        _render_swarm_workflow()
    elif workflow_type == "Codebase Analyst":
        _render_code_analysis_workflow()

def _render_code_analysis_workflow() -> None:
    """Render the Code Analysis workflow."""
    st.markdown("#### 💻 Codebase Analyst")
    st.markdown("An agent that can read your local files and suggest refactoring or improvements.")
    
    # Simple File Selector (Mock)
    import os
    files = [f for f in os.listdir(".") if f.endswith(".py") or f.endswith(".md")]
    selected_file = st.selectbox("Select File to Analyze", files)
    
    prompt = st.text_area("Analysis Goal", "Explain this code and suggest 3 improvements.")
    
    if st.button("🔍 Analyze Code"):
        try:
            with open(selected_file, "r") as f:
                content = f.read()
            
            # Simple direct LLM call for now
            # In a real swarm, this would be: [Ingest] -> [Reviewer] -> [Security] -> [Refactorer]
            st.info(f"Analyzing {selected_file}...")
            
            from core.llm_client import LLMClient
            client = LLMClient(provider="gemini") # Default to Gemini
            
            analysis_prompt = f"""
            You are a Senior Software Engineer. Analyze the following file: '{selected_file}'.
            
            GOAL: {prompt}
            
            CODE CONTENT:
            ```
            {content}
            ```
            """
            
            response = client.generate_text(analysis_prompt)
            st.markdown("### 🛠️ Analyst Report")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

def _render_swarm_workflow() -> None:
    """Render the Strategic Swarm workflow (Calling FastAPI Backend)."""
    st.markdown("#### 🐝 Strategic Swarm (Multi-Agent Parallel)")
    st.markdown("This workflow triggers a **Planner** who fans out to **Market**, **Tech**, and **Risk** analysts in parallel.")
    
    topic = st.text_input("Enter Topic for Analysis", "The future of Agentic AI in healthcare")
    
    if st.button("🚀 Launch Swarm"):
        # We call the FastAPI backend here to demonstrate decoupling
        import requests
        
        with st.spinner("Swarm agents are collaborating via FastAPI backend..."):
            try:
                # Assuming backend runs on port 8000 (standard in our docker-compose)
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                
                payload = {
                    "task": topic,
                    "provider": "gemini" # Default
                }
                
                response = requests.post(f"{backend_url}/swarm/run", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ Swarm Analysis Complete!")
                    
                    # Layout Results
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
                    st.warning("Ensure the FastAPI server is running (`uvicorn api.main:app --reload`) or use the local fallback.")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.info("Falling back to local execution for demonstration...")
                
                # Fallback to local if backend is not reachable
                from core.agents import create_swarm_graph
                graph = create_swarm_graph()
                result = graph.invoke({"topic": topic})
                st.markdown(result.get("final_report", "Error"))

def _render_research_workflow() -> None:
    """Render the Research & Report workflow interface."""
    st.markdown("#### 🕵️ Research Team (Researcher + Writer)")
    st.markdown("A multi-agent system where a **Researcher** gathers facts and a **Writer** creates a blog post.")
    
    # Inputs
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
        
        # Initialize Graph
        try:
            graph = create_research_graph(provider=provider)
        except Exception as e:
            st.error(f"Failed to initialize graph: {e}")
            return
            
        # UI Layout
        col_status, col_graph = st.columns([1, 1])
        
        with col_graph:
            with st.expander("View Graph Topology", expanded=True):
                try:
                    graph_img = graph.get_graph().draw_mermaid_png()
                    st.image(graph_img, caption="Multi-Agent Workflow", use_container_width=True)
                except Exception:
                    st.warning("Could not generate graph image.")

        with col_status:
            # Session State for Thread
            if "research_thread_id" not in st.session_state:
                st.session_state.research_thread_id = "thread_1" # Simple fixed thread for demo
            
            thread_config = {"configurable": {"thread_id": st.session_state.research_thread_id}}
            
            st.info(f"🔄 **Team Active**: Researcing '{topic}'...")
            status_container = st.empty()
            
            # 1. Start execution or Resume
            if st.button("▶️ Run / Resume Workflow"):
                status_container.text("Agents working...")
                try:
                    # Initial Run
                    current_state = graph.get_state(thread_config)
                    
                    if not current_state.values:
                         # Start fresh
                         graph.invoke(
                             {"messages": [("user", f"Research this topic: {topic}")]}, 
                             thread_config
                         )
                    else:
                        # Resume (if we were paused)
                        # We just call invoke with None to resume from interrupt
                        graph.invoke(None, thread_config)
                        
                except Exception as e:
                    # st.error(f"Execution Error: {e}")
                    pass
            
            # 2. Check State
            state_snapshot = graph.get_state(thread_config)
            
            if state_snapshot.next:
                # If next node is 'manager', we are PAUSED
                if "manager" in state_snapshot.next:
                    st.warning("⚠️ **Manager Review Required**")
                    
                    # Show latest findings
                    last_msg = state_snapshot.values["messages"][-1]
                    st.markdown("### 🔎 Researcher Findings")
                    st.markdown(last_msg.content)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Approve"):
                            graph.update_state(thread_config, {"messages": [HumanMessage(content="APPROVE")]}, as_node="manager")
                            st.rerun() # Rerun to hit 'Resume' logic naturally or auto-trigger? 
                                       # Better to let user click Resume or auto-resume. 
                                       # For simplicity, we just update state, user clicks Run/Resume.
                            st.success("Approved! Click 'Run/Resume' to proceed.")

                    with c2:
                        feedback = st.text_input("Feedback for Rejection")
                        if st.button("↩️ Reject"):
                             graph.update_state(thread_config, {"messages": [HumanMessage(content=f"REJECT: {feedback}")]}, as_node="manager")
                             st.error("Rejected sent. Click 'Run/Resume' to retry.")

            # 3. Completion Check
            # If no next, we are done
            if not state_snapshot.next and state_snapshot.values:
                 final_msg = state_snapshot.values["messages"][-1]
                 status_container.success("✅ Work Complete!")
                 st.markdown("### 📄 Final Report")
                 st.markdown(final_msg.content)

