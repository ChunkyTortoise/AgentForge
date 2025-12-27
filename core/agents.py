"""
Agent Framework - LangGraph Implementation.

Demonstrates:
- LangGraph application structure
- State management
- Tool integration
- Unified LLM access via LLMClient
"""
import operator
from typing import Annotated, Sequence, TypedDict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup

from core.llm_client import LLMClient
from core.config import settings
from core.tools import list_files, read_file
from utils.logger import get_logger

logger = get_logger(__name__)


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
        text = soup.get_text(separator="\n")
        return text[:8000]  # Limit context
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

# --- NODES ---

class AgentState(TypedDict):
    """The state of the agent in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_data: str  # Structure to hold gathered research

def create_research_graph(provider: str = None):
    """
    Creates a Multi-Agent Research Graph:
    [Researcher] -> [Writer]
    """
    # Lazy import to avoid dependency issues if not used
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    
    client = LLMClient(provider=provider)
    llm = client.get_langchain_model()
    
    # 1. Researcher Node
    researcher_llm = llm.bind_tools([search_tool, scrape_web_page])
    
    def researcher_node(state: AgentState):
        """Conducts research using tools."""
        messages = state["messages"]
        response = researcher_llm.invoke(messages)
        return {"messages": [response]}

    def manager_node(state: AgentState):
        """Placeholder for human-in-the-loop review."""
        pass

    # 2. Writer Node
    writer_llm = llm
    
    def writer_node(state: AgentState):
        """Synthesize research into a post."""
        messages = state["messages"]
        prompt = "You are a Professional Tech Writer. Using the above research context, write a comprehensive, engaging blog post. Use Markdown formatting."
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

    def researcher_router(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "manager"

    workflow.add_conditional_edges("researcher", researcher_router)
    workflow.add_edge("tools", "researcher")
    
    def manager_router(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage):
             if "APPROVE" in last_message.content.upper():
                 return "writer"
             else:
                 return "researcher"
        
        return "writer"

    workflow.add_conditional_edges(
        "manager",
        manager_router,
        {"writer": "writer", "researcher": "researcher"}
    )
    
    workflow.add_edge("writer", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["manager"])


# --- SWARM IMPLEMENTATION ---

class SwarmState(TypedDict):
    """State for the parallel swarm."""
    topic: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    analyst_outputs: Annotated[list[str], operator.add]
    final_report: str


def create_swarm_graph(provider: str = None):
    """
    Creates a Parallel Swarm Graph:
    Planner -> [Market, Tech, Risk] (Parallel) -> Aggregator
    """
    client = LLMClient(provider=provider)
    llm = client.get_langchain_model()

    def planner_node(state: SwarmState):
        return {"messages": [AIMessage(content=f"Initiating swarm analysis for: {state['topic']}")]}

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

    def aggregator_node(state: SwarmState):
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

    workflow = StateGraph(SwarmState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("tech_analyst", tech_analyst_node)
    workflow.add_node("risk_analyst", risk_analyst_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "market_analyst")
    workflow.add_edge("planner", "tech_analyst")
    workflow.add_edge("planner", "risk_analyst")

    workflow.add_edge("market_analyst", "aggregator")
    workflow.add_edge("tech_analyst", "aggregator")
    workflow.add_edge("risk_analyst", "aggregator")

    workflow.add_edge("aggregator", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# --- TODO SOLVER IMPLEMENTATION ---

class TodoState(TypedDict):
    """State for the TODO Solver Agent."""
    target_file: str # The TODO file
    messages: Annotated[Sequence[BaseMessage], operator.add]
    selected_task: str
    code_proposal: str

def create_todo_solver_graph(provider: str = None):
    """
    Creates a TODO Solver Graph:
    [Parser] -> [ContextFinder (Tool Loop)] -> [Architect]
    """
    client = LLMClient(provider=provider)
    llm = client.get_langchain_model()
    
    # 1. Task Parser Node (Reads TODO.md)
    def parser_node(state: TodoState):
        """Reads TODO file and picks the first unchecked task."""
        logger.info(f"Scanning {state['target_file']}...")
        
        # We manually read the file here to bootstrap the process
        # In a purist agent, we'd use a tool, but this is an initialization step
        try:
            with open(state["target_file"], "r") as f:
                content = f.read()
            
            # Simple heuristic: Find first line with "- [ ]" 
            import re
            match = re.search(r"- \[ \] (.*)", content)
            if match:
                task = match.group(1).strip()
                return {
                    "selected_task": task,
                    "messages": [AIMessage(content=f"I have identified the next task: '{task}'. I will now investigate the codebase.")]
                }
            else:
                return {
                    "selected_task": "None",
                    "messages": [AIMessage(content="No pending tasks found in TODO.md.")]
                }
        except Exception as e:
            return {
                "selected_task": "Error",
                "messages": [AIMessage(content=f"Error reading TODO file: {e}")]
            }

    # 2. Context Finder Node (Uses Tools)
    # Give the LLM tools to explore files
    context_llm = llm.bind_tools([list_files, read_file])
    
    def context_finder_node(state: TodoState):
        """Explores the codebase to find relevant context."""
        if state["selected_task"] in ["None", "Error"]:
            return {"messages": [AIMessage(content="Stopping execution.")]}
            
        messages = state["messages"]
        last_msg = messages[-1]
        
        # Injection to guide the agent if it's the first step after parser
        if len(messages) == 1:
            prompt = f"""
            You are a Senior Engineer tasked with solving this TODO: "{state['selected_task']}".
            
            Your goal is to:
            1. Understand the current codebase structure (use 'list_files').
            2. Read relevant files to understand where to make changes (use 'read_file').
            3. Once you have enough context, stop calling tools.
            """
            response = context_llm.invoke([HumanMessage(content=prompt)])
        else:
            response = context_llm.invoke(messages)
            
        return {"messages": [response]}

    # 3. Tool Node
    tool_node = ToolNode([list_files, read_file])

    # 4. Architect Node
    def architect_node(state: TodoState):
        """Proposes a code solution."""
        messages = state["messages"]
        
        prompt = f"""
        Based on the context you have gathered, propose a detailed code solution for the task: "{state['selected_task']}".
        
        Provide the response in this format:
        ## Plan
        [Steps]
        
        ## Code Changes
        ```python
        # File: path/to/file.py
        [Code snippet]
        ```
        """
        response = llm.invoke(messages + [HumanMessage(content=prompt)])
        return {"code_proposal": response.content}

    # Graph
    workflow = StateGraph(TodoState)
    
    workflow.add_node("parser", parser_node)
    workflow.add_node("context_finder", context_finder_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("architect", architect_node)
    
    workflow.set_entry_point("parser")
    
    # Edges
    workflow.add_edge("parser", "context_finder")
    
    def finder_router(state: TodoState):
        if state["selected_task"] in ["None", "Error"]:
            return END
            
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "architect"
        
    workflow.add_conditional_edges("context_finder", finder_router)
    workflow.add_edge("tools", "context_finder")
    workflow.add_edge("architect", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
