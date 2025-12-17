"""
Smart Analyst Module - Generative BI & Data Analytics.

Demonstrates:
- Unified Intelligence Engine (Agents + Tools)
- Google Data Analytics (Data Analysis)
- IBM RAG (Context-aware Analysis)
- LangChain Pandas Agent
"""
import os
import pandas as pd
import streamlit as st
import plotly.express as px

# Import our Core components
from core.rag_engine import VectorStore
from utils.logger import get_logger
from core.llm_client import get_available_providers
from langchain_core.tools import Tool
from langchain_experimental.agents import create_pandas_dataframe_agent

logger = get_logger(__name__)

def render() -> None:
    """Render the Smart Analyst module."""
    st.markdown("## 📊 Smart Analyst: Generative BI")
    st.markdown("""
    Combine **Data Analysis** with **Business Context** to generate strategic insights.
    """)

    # 1. State Management
    if "analyst_agent" not in st.session_state:
        st.session_state.analyst_agent = None
    if "analyst_df" not in st.session_state:
        st.session_state.analyst_df = None
    if "analyst_messages" not in st.session_state:
        st.session_state.analyst_messages = []

    # Check APIs
    providers = get_available_providers()
    if not any(providers.values()):
        st.error("⚠️ No LLM API keys found. Please check your .env file.")
        return

    # 2. Main Layout - Tabs
    tab_dashboard, tab_analysis, tab_config = st.tabs(["📈 Dashboard", "💬 Analysis Chat", "⚙️ Configuration"])

    # --- CONFIGURATION TAB ---
    with tab_config:
        st.subheader("Data & Context Setup")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Data Source")
            uploaded_csv = st.file_uploader("Upload Data (CSV)", type="csv", key="csv_upload")
            use_demo = st.button("Load Demo Dataset")
            
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.session_state.analyst_df = df
                    st.success(f"loaded {len(df)} rows from CSV.")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
            
            if use_demo:
                st.session_state.analyst_df = _get_demo_data()
                st.success("Loaded Demo Dataset (Sales Data).")

        with col2:
            st.markdown("### 2. Business Context (Optional)")
            uploaded_pdf = st.file_uploader("Upload Definitions/Context (PDF)", type="pdf", key="pdf_upload")
            
            if uploaded_pdf and "analyst_vector_store" not in st.session_state:
                 with st.spinner("Indexing Context..."):
                    st.session_state.analyst_vector_store = _process_context(uploaded_pdf)
                    st.success("Context Indexed!")

        # Data Preview
        if st.session_state.analyst_df is not None:
            st.divider()
            st.markdown("### Dataset Preview")
            st.dataframe(st.session_state.analyst_df.head(), use_container_width=True)

    # --- DASHBOARD TAB ---
    with tab_dashboard:
        if st.session_state.analyst_df is None:
            st.info("👈 Please load data in the 'Configuration' tab to view the dashboard.")
        else:
            st.subheader("Quick Insights")
            
            # Auto-generated metrics (Simple Logic)
            df = st.session_state.analyst_df
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_cols) > 0:
                cols = st.columns(min(len(numeric_cols), 4))
                for i, col in enumerate(numeric_cols[:4]):
                    cols[i].metric(label=col, value=f"{df[col].mean():.2f}")
            
            st.divider()
            st.markdown("**Automated Analysis (Experimental)**")
            if st.button("Generate Autoreport"):
                 with st.spinner("Analyzing..."):
                     # Simple heuristics for demo
                     summary = df.describe().to_markdown()
                     st.text(summary)

    # --- ANALYSIS CHAT TAB ---
    with tab_analysis:
        if st.session_state.analyst_df is None:
            st.info("Please load data first.")
        else:
            # Chat Container
            chat_container = st.container()
            
            # Input
            query = st.chat_input("Ask a question (e.g., 'Plot sales by Region', 'Who is the top customer?')")
            
            # Render History
            with chat_container:
                for msg in st.session_state.analyst_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if "plot" in msg:
                            st.plotly_chart(msg["plot"], use_container_width=True)
            
            # Processing
            if query:
                # User Msg
                st.session_state.analyst_messages.append({"role": "user", "content": query})
                with chat_container:
                     with st.chat_message("user"):
                        st.markdown(query)
                     
                     with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                             response, plot_fig = _run_analysis_agent(
                                 query,
                                 st.session_state.analyst_df,
                                 st.session_state.get("analyst_vector_store")
                             )
                             
                             st.markdown(response)
                             if plot_fig:
                                 st.plotly_chart(plot_fig, use_container_width=True)
                             
                             # Save State
                             msg_data = {"role": "assistant", "content": response}
                             if plot_fig:
                                 msg_data["plot"] = plot_fig
                             st.session_state.analyst_messages.append(msg_data)


def _get_demo_data() -> pd.DataFrame:
    """Generate rich demo data."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        "Date": dates,
        "Region": np.random.choice(["North", "South", "East", "West"], 100),
        "Product": np.random.choice(["Laptop", "Mouse", "Monitor", "Keyboard"], 100),
        "Sales": np.random.randint(100, 1000, 100),
        "Units": np.random.randint(1, 50, 100),
        "Customer_Satisfaction": np.random.randint(1, 6, 100)
    }
    return pd.DataFrame(data)

def _process_context(pdf_file) -> VectorStore:
    """Process PDF into a temporary vector store."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        logger.error(f"PDF Error: {e}")
        text = "Error reading PDF"

    # Create Store
    vs = VectorStore(
        collection_name=f"temp_{pdf_file.name}", 
        persist_directory=f"./.chroma_temp/{pdf_file.name}"
    )
    # Chunking
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    vs.add_texts(chunks)
    return vs

def _run_analysis_agent(query: str, df: pd.DataFrame, vector_store: VectorStore = None):
    """
    # Run the LangChain Pandas Agent with optional RAG tools.
    """
    # Prefer Claude for Code Generation if available
    if os.getenv("ANTHROPIC_API_KEY"):
         from langchain_anthropic import ChatAnthropic
         llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

    # Define Tools
    extra_tools = []
    generated_plots = []
    
    def visualize_data(x_col: str, y_col: str, chart_type: str = "line", title: str = None) -> str:
        """
        Create a Plotly chart. 
        Args:
            x_col: Column name for X axis
            y_col: Column name for Y axis
            chart_type: 'line', 'bar', 'scatter', 'histogram', 'box', 'pie', 'area'
            title: Chart title
        """
        try:
            # Verify columns
            if x_col not in df.columns:
                return f"Error: Column '{x_col}' not found. Available: {list(df.columns)}"
            # y_col can be optional for some plots like histogram (count) or pie
            if y_col and y_col not in df.columns:
                return f"Error: Column '{y_col}' not found. Available: {list(df.columns)}"
            
            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_col, y=y_col, title=title)
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            elif chart_type == "area":
                fig = px.area(df, x=x_col, y=y_col, title=title)
            else:
                return f"Error: Unsupported chart type {chart_type}. Use line, bar, scatter, histogram, box, pie, area."
                
            fig.update_layout(template="plotly_white")
            generated_plots.append(fig)
            return "Chart created successfully."
        except Exception as e:
            return f"Error creating chart: {e}"

    extra_tools.append(Tool(
        name="visualize_data",
        func=visualize_data,
        description="Create charts. Inputs: x_col, y_col, chart_type ('line', 'bar', 'scatter', 'box', 'pie', 'area'), title."
    ))

    if vector_store:
        def context_search(q: str):
            """Search business context/definitions."""
            try:
                results = vector_store.search(q, n_results=2)
                return "\\n".join([r.text for r in results])
            except Exception:
                return "No context found."
        
        extra_tools.append(Tool(
            name="lookup_business_context",
            func=context_search,
            description="Search for business definitions or context in the uploaded PDF."
        ))

    # Create Agent
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        agent_type="tool-calling",
        extra_tools=extra_tools,
        allow_dangerous_code=True,
        max_iterations=5,
        include_df_in_prompt=True,
        number_of_head_rows=5
    )

    # Retry Loop for Self-Healing
    max_retries = 3
    current_try = 0
    last_error = None
    
    while current_try < max_retries:
        try:
            full_prompt = f"""
            You are a Strategic Data Analyst. 
            
            Tasks:
            1. Analyze the dataframe `df`.
            2. If specific business terms are used, check 'lookup_business_context'.
            3. CREATE CHARTS using 'visualize_data' whenever trends, comparisons, or distributions are asked for.
            4. If a user asks for a plot but the columns don't match, Infer the closest valid column from the dataframe preview.
            
            Question: {query}
            """
            
            if last_error:
                full_prompt += f"\\n\\n⚠️ PREVIOUS ERROR: {last_error}\\n"
                full_prompt += "Fix the error and retry. Check column names carefully."
            
            result = agent.invoke(full_prompt)
            response_text = result['output']
            
            plot_fig = generated_plots[-1] if generated_plots else None
            
            return response_text, plot_fig

        except Exception as e:
            logger.warning(f"Agent Attempt {current_try + 1} Failed: {e}")
            last_error = str(e)
            current_try += 1
            
    # If all retries fail
    return f"I encountered an issue analyzing the data: {last_error}", None

