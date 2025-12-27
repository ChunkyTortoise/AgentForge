import streamlit as st
from dotenv import load_dotenv
import os
from utils.logger import get_logger

# Page Config (MUST BE FIRST)
st.set_page_config(
    page_title="AgentForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Logger
logger = get_logger(__name__)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/css/style.css")
except FileNotFoundError:
    pass

# Load Environment Variables
load_dotenv()

# Module registry
MODULES = {
    "🔍 RAG Assistant": ("rag_assistant", "RAG Document Assistant"),
    "🤖 Agent Hub": ("agent_hub", "Multi-Agent Research Hub"),
    "🧪 Evaluation Lab": ("evaluation_lab", "System Performance Lab"),
    "📊 Smart Analyst": ("smart_analyst", "AI-Enhanced Data Analytics"),
    "👁️ Vision Forge": ("vision_forge", "Multimodal Intelligence"),
    "⚡ Prompt Lab": ("prompt_lab", "Prompt Engineering Tools"),
    "️ Model Arena": ("model_arena", "LLM Comparison"),
}


def main() -> None:
    """Main application function."""
    try:
        # Sidebar navigation
        st.sidebar.title("🔮 AgentForge")
        st.sidebar.markdown("**AI Intelligence Platform**")
        
        # API Health Check
        import requests
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        try:
            api_status = requests.get(f"{backend_url}/", timeout=1).status_code
            if api_status == 200:
                st.sidebar.success("● API Online")
        except:
            st.sidebar.error("○ API Offline (Local Fallback)")
            
        st.sidebar.markdown("---")

        # Navigation
        pages = ["🏠 Overview"] + list(MODULES.keys())
        page = st.sidebar.radio("Navigate:", pages)

        logger.info(f"User navigated to: {page}")

        # Render page
        if page == "🏠 Overview":
            _render_overview()
        elif page in MODULES:
            module_name, title = MODULES[page]
            _render_module(module_name, title)

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error("An unexpected error occurred.")
        if st.checkbox("Show error details"):
            st.exception(e)


def _render_overview() -> None:
    """Render the overview/home page."""
    st.title("🔮 AgentForge")
    st.markdown("### AI-Powered Intelligence Platform")
    st.markdown("""
    Welcome to AgentForge - a demonstration of production-grade AI/ML engineering
    featuring RAG, multi-agent systems, and AI-enhanced analytics.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modules", f"{len(MODULES)}", delta="Active")
    with col2:
        st.metric("LLM Providers", "2", delta="Gemini + Claude")
    with col3:
        st.metric("Certifications", "19", delta="1,768+ hours")
    with col4:
        st.metric("Focus", "AI/ML", delta="Production-Grade")

    st.markdown("---")

    # Module cards
    st.markdown("### 🚀 Available Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔍 RAG Assistant
        Upload documents and ask questions. Get AI-powered answers with
        source citations. Built with LangChain and ChromaDB.

        **Skills**: IBM RAG/Agentic AI, Duke LLMOps

        ---

        #### 📊 Smart Analyst
        Explore data using natural language. AI generates insights,
        visualizations, and SQL queries automatically.

        **Skills**: Google Data Analytics, Microsoft GenAI for Data

        ---

        #### 🏟️ Model Arena
        Compare LLM responses side-by-side. Evaluate quality, speed,
        and cost across Gemini and Claude.

        **Skills**: Duke LLMOps, DeepLearning.AI
        """)

    with col2:
        st.markdown("""
        #### 🤖 Agent Hub
        Create multi-agent workflows for research and analysis.
        Agents collaborate to complete complex tasks autonomously.

        **Skills**: Google Cloud GenAI Agents, IBM Agentic AI

        ---

        #### ⚡ Prompt Lab
        Test and optimize prompts. A/B testing, token analysis,
        and template library for production use.

        **Skills**: Vanderbilt Prompt Engineering

        ---

        #### 🎓 Certifications
        This project showcases **1,768+ hours** of professional learning
        across AI/ML, Data Analytics, and GenAI engineering.
        """)

    st.markdown("---")

    # Tech stack
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    | Category | Technologies |
    |----------|-------------|
    | **LLMs** | Google Gemini, Anthropic Claude |
    | **AI Framework** | LangChain, LangGraph |
    | **Vector DB** | ChromaDB, FAISS |
    | **Embeddings** | sentence-transformers |
    | **UI** | Streamlit, Plotly |
    | **Data** | Pandas, NumPy |
    """)


def _render_module(module_name: str, title: str) -> None:
    """Render a module page."""
    st.title(title)
    try:
        import importlib
        module = importlib.import_module(f"modules.{module_name}")
        module.render()
    except ModuleNotFoundError:
        st.warning(f"Module '{module_name}' is under development.")
        st.info("Check back soon for this feature!")
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {e}")
        st.error(f"Failed to load {title}")
        if st.checkbox("Show details", key=f"{module_name}_error"):
            st.exception(e)


if __name__ == "__main__":
    main()
