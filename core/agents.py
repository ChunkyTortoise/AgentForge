"""
Agent Framework - LangGraph Implementation.

Demonstrates:
- LangGraph application structure
- State management
- Tool integration
- Multi-provider support (Gemini/Claude via LangChain)
"""
import os
import operator
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class AgentState(TypedDict):
    """The state of the agent in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_data: str  # Structure to hold gathered research


class BaseAgent:
    """Base Agent wrapper for LangChain models."""
    def __init__(self, provider: str, model: str = None, name: str = "Agent"):
        self.name = name
        self.llm = self._get_llm(provider, model)

    def _get_llm(self, provider: str, model: str) -> BaseChatModel:
        if provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found. Please set it in .env")
            model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name, google_api_key=api_key, temperature=0.0, convert_system_message_to_human=True
            )
        elif provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found. Please set it in .env")
            model_name = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            return ChatAnthropic(
                model=model_name, anthropic_api_key=api_key, temperature=0.0
            )
        raise ValueError(f"Unsupported provider: {provider}")


# --- TOOLS ---

@tool
def scrape_web_page(url: str) -> str:
    """Scrape the content of a web page URL for deep details."""
    try:
        headers = {"User-Agent": "AgentForge/1.0 (Research Bot)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Cleanup
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text = soup.get_text(separator="\\n")
        return text[:8000]  # Limit context
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

search_tool = DuckDuckGoSearchRun()


# --- NODES ---

def create_research_graph(provider: str = "gemini"):
    """
    Creates a Multi-Agent Research Graph:
    [Researcher] -> [Writer]
    """
    
    # 1. Researcher Node
    researcher_llm = BaseAgent(provider).llm.bind_tools([search_tool, scrape_web_page])
    
    def researcher_node(state: AgentState):
        """Conducts research using tools."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If tool output, we continue research or summarize
        if isinstance(last_message, FunctionMessage) or (hasattr(last_message, 'tool_calls') and not last_message.tool_calls):
             # It's a response after tool execution
             # We can assume research is done for this turn
             pass
             
        # Prompt injection for role
        if len(messages) == 1:
            system_msg = BaseMessage(content="You are a Lead Researcher. Search for detailed information on the user's topic. Use 'scrape_web_page' to get details. Summarize findings into 'RESEARCH_DATA'.", type="system")
            # Note: simplistic injection, ideally use SystemMessage
            # But we'll just rely on the prompt in invoke
            pass

        response = researcher_llm.invoke(messages)
        return {"messages": [response]}

    # 1.5 Manager Node (Human-in-the-Loop)
    def manager_node(state: AgentState):
        """
        Manager review node. 
        This is a placeholder that does nothing but pause the graph 
        if we use 'interrupt_before'. 
        Or we can check for state updates injected by the human.
        """
        # In a real HITL system, we might look for a specific 'feedback' message
        pass

    # 2. Writer Node
    writer_llm = BaseAgent(provider).llm
    
    def writer_node(state: AgentState):
        """Synthesize research into a post."""
        messages = state["messages"]
        # Filter for the research summary or last AI message
        # In a real system we'd parse 'research_data' from state
        # For simplicity, we pass the entire conversation history
        
        prompt = "You are a Professional Tech Writer. Using the above research context, write a comprehensive, engaging blog post. Use Markdown formatting."
        
        # We append a human message with the instruction
        response = writer_llm.invoke(messages + [HumanMessage(content=prompt)])
        return {"messages": [response]}

    # 3. Tool Node
    tool_node = ToolNode([search_tool, scrape_web_page])

    # 4. Graph Construction
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("manager", manager_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("researcher")

    # Conditional logic for Researcher -> Tools or Manager
    def researcher_router(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "manager"

    workflow.add_conditional_edges("researcher", researcher_router)
    workflow.add_edge("tools", "researcher") # Loop back to researcher
    
    # Manager Router
    def manager_router(state: AgentState):
        # This logic depends on Human interaction.
        # If the last message is from 'human' and says "APPROVE", go to writer.
        # If "REJECT", go back to researcher.
        
        messages = state["messages"]
        last_message = messages[-1]
        
        # Simple heuristic: If the LAST message is Human (feedback), we check content
        if isinstance(last_message, HumanMessage):
             if "APPROVE" in last_message.content.upper():
                 return "writer"
             else:
                 return "researcher"
        
        # Default: If we just arrived here from Researcher, we pause. 
        # But we need to return something to keep graph valid?
        # Actually, if we use interrupt_before=["manager"], execution stops BEFORE running manager.
        # When we resume, we are IN manager. 
        
        return "writer" # Default fallback if no feedback mechanism logic yet

    workflow.add_conditional_edges(
        "manager",
        manager_router,
        {"writer": "writer", "researcher": "researcher"}
    )
    
    workflow.add_edge("writer", END)
    
    # Add Persistence
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory, interrupt_before=["manager"])


# --- SWARM IMPLEMENTATION ---

class SwarmState(TypedDict):
    """State for the parallel swarm."""
    topic: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    analyst_outputs: Annotated[list[str], operator.add]
    final_report: str


def create_swarm_graph(provider: str = "gemini"):
    """
    Creates a Parallel Swarm Graph:
    Planner -> [Market, Tech, Risk] (Parallel) -> Aggregator
    """
    llm = BaseAgent(provider).llm

    # 1. Planner Node
    def planner_node(state: SwarmState):
        """Just initializes the workflow."""
        return {"messages": [AIMessage(content=f"Initiating swarm analysis for: {state['topic']}")]}

    # 2. Parallel Analysts
    def market_analyst_node(state: SwarmState):
        prompt = f"You are a Market Analyst. Analyze the market potential, trends, and target audience for: '{state['topic']}'. Be concise (3 bullets)."
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analyst_outputs": [f"### 📊 Market Analysis\n{response.content}"]}

    def tech_analyst_node(state: SwarmState):
        prompt = f"You are a Technology Expert. Analyze the technical feasibility, stack, and innovation for: '{state['topic']}'. Be concise (3 bullets)."
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analyst_outputs": [f"### 💻 Technical Analysis\n{response.content}"]}

    def risk_analyst_node(state: SwarmState):
        prompt = f"You are a Risk Officer. Analyze potential legal, ethical, and operational risks for: '{state['topic']}'. Be concise (3 bullets)."
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analyst_outputs": [f"### 🛡️ Risk Assessment\n{response.content}"]}

    # 3. Aggregator
    def aggregator_node(state: SwarmState):
        """Synthesize all parallel outputs."""
        outputs = "\n\n".join(state["analyst_outputs"])
        prompt = f"""
        You are a Lead Strategist. Synthesize the following swarm reports into a cohesive executive summary about '{state['topic']}'.
        
        SWARM REPORTS:
        {outputs}
        
        Final Report format:
        # Executive Strategy: {state['topic']}
        ## Executive Summary
        [Synthesized content]
        ## Key Insights
        [Bullets]
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": response.content}

    # 4. Graph Construction
    workflow = StateGraph(SwarmState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("tech_analyst", tech_analyst_node)
    workflow.add_node("risk_analyst", risk_analyst_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.set_entry_point("planner")

    # Fan-out: Planner -> All Analysts
    workflow.add_edge("planner", "market_analyst")
    workflow.add_edge("planner", "tech_analyst")
    workflow.add_edge("planner", "risk_analyst")

    # Fan-in: All Analysts -> Aggregator
    workflow.add_edge("market_analyst", "aggregator")
    workflow.add_edge("tech_analyst", "aggregator")
    workflow.add_edge("risk_analyst", "aggregator")

    workflow.add_edge("aggregator", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
