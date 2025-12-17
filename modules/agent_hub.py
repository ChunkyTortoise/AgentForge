"""
Agent Hub Module - Multi-Agent Research System.

Demonstrates:
- IBM RAG and Agentic AI
- Google Cloud GenAI Agents
- LangGraph and CrewAI concepts
"""
import streamlit as st

from core.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Agent Hub module."""
    st.markdown("""
    # Workflow Selection
    """)
    st.markdown("### 🛠️ Agent Workflows")
    
    workflow_type = st.selectbox(
        "Select Workflow",
        ["Research & Report", "Code Analysis (Coming Soon)", "Data Pipeline (Coming Soon)"]
    )
    
    if workflow_type == "Research & Report":
        _render_research_workflow()

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
            st.info(f"🔄 **Team Active**: Researcing '{topic}'...")
            status_container = st.empty()
            status_container.text("Analyzing request...")
            
            # Execute
            try:
                # We can stream events later, for now just run
                inputs = {"messages": [("user", f"Research this topic: {topic}")]}
                result = graph.invoke(inputs)
                
                final_msg = result["messages"][-1]
                status_container.success("✅ Work Complete!")
                
                st.markdown("### 📄 Final Report")
                st.markdown(final_msg.content)
                
            except Exception as e:
                status_container.error(f"Workflow failed: {str(e)}")
