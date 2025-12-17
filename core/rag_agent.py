"""
Agentic RAG Module - Self-Corrective Retrieval Augmented Generation.

Logic:
1. Retrieve documents using VectorStore.
2. Grade relevance of documents using LLM.
3. If irrelevant -> Rewrite Query and loop.
4. If relevant -> Generate Answer.
"""
from typing import TypedDict, List
import os

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, END

from utils.logger import get_logger

logger = get_logger(__name__)


# --- STATE ---
class RAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retry_count: int


# --- DATA MODELS ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# --- NODE FUNCTIONS ---

def create_rag_graph(db_retriever, provider="gemini"):
    """
    Factory to create the RAG Graph.
    Args:
        db_retriever: Function or tool to retrieve docs (VectorStore.search)
        provider: 'gemini' or 'claude'
    """
    
    # 1. Setup LLM
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

    # 2. Nodes
    
    def retrieve(state: RAGState):
        """Node: Retrieve documents."""
        print(f"---RETRIEVE--- Query: {state['question']}")
        documents = db_retriever(state['question']) # Assumes returns list of strings
        # Format if necessary (VectorStore returns objects)
        if documents and not isinstance(documents[0], str): 
             # Simplify object to string
             documents = [f"{d.text or ''} (Source: {d.source})" for d in documents]
             
        return {"documents": documents, "question": state['question']}

    def grade_documents(state: RAGState):
        """Node: Grade relevance."""
        print("---CHECK RELEVANCE---")
        question = state['question']
        documents = state['documents']
        
        # Grading Chain
        structured_llm = llm.with_structured_output(GradeDocuments)
        
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        chain = prompt | structured_llm
        
        # Grade inputs
        filtered_docs = []
        for d in documents:
            score = chain.invoke({"question": question, "context": d})
            if score.binary_score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                
        return {"documents": filtered_docs, "question": question}

    def transform_query(state: RAGState):
        """Node: Rewrite query."""
        print("---TRANSFORM QUERY---")
        question = state['question']
        
        # Rewriter
        msg = [
            HumanMessage(content=f"""Look at the input and try to reason about the underlying, semantic intent / meaning. \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Formulate an improved question: """), 
        ]
        
        response = llm.invoke(msg)
        return {"question": response.content, "retry_count": state.get("retry_count", 0) + 1}

    def generate(state: RAGState):
        """Node: Generate answer."""
        print("---GENERATE---")
        question = state['question']
        documents = state['documents']
        
        # RAG Chain
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:""",
            input_variables=["question", "context"],
        )
        
        # Allow LLM to just ingest text
        chain = prompt | llm | StrOutputParser()
        
        generation = chain.invoke({"context": "\n\n".join(documents), "question": question})
        return {"generation": generation}

    # 3. Conditional Edges
    def decide_to_generate(state: RAGState):
        """Edge: Re-grade."""
        filtered_documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        
        if not filtered_documents:
            # All docs filtered out
            if retry_count > 1:
                # Give up to avoid infinite loop -> Go to Final Answer anyway (or failure state)
                return "generate" 
            return "transform_query"
        else:
            return "generate"

    # 4. Build Graph
    workflow = StateGraph(RAGState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
