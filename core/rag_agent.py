from typing import TypedDict, List

    from langchain_core.messages import HumanMessage
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from langgraph.graph import StateGraph, END

    from core.llm_client import LLMClient
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

def create_rag_graph(db_retriever, provider=None):
    """
    Factory to create the RAG Graph.
    Args:
        db_retriever: Function or tool to retrieve docs (VectorStore.search)
        provider: 'gemini' or 'claude'
    """
    
    # 1. Setup LLM via Unified Client
    client = LLMClient(provider=provider)
    llm = client.get_langchain_model(temperature=0)

    # 2. Nodes
    
    def retrieve(state: RAGState):
        """Node: Retrieve documents."""
        logger.info(f"---RETRIEVE--- Query: {state['question']}")
        documents = db_retriever(state['question']) # Assumes returns list of strings
        # Format if necessary (VectorStore returns objects)
        if documents and not isinstance(documents[0], str): 
             # Simplify object to string
             documents = [f"{getattr(d, 'page_content', str(d))} (Source: {getattr(d, 'metadata', {}).get('source', 'unknown')})" for d in documents]
             
        return {"documents": documents, "question": state['question']}

    def grade_documents(state: RAGState):
        """Node: Grade relevance."""
        logger.info("---CHECK RELEVANCE---")
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
            try:
                score = chain.invoke({"question": question, "context": d})
                if score.binary_score == "yes":
                    logger.info("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            except Exception as e:
                logger.error(f"Error grading document: {e}")
                # Fallback to keep doc if grading fails
                filtered_docs.append(d)
                
        return {"documents": filtered_docs, "question": question}

    def transform_query(state: RAGState):
        """Node: Rewrite query."""
        logger.info("---TRANSFORM QUERY---")
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
        logger.info("---GENERATE---")
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
            if retry_count > 1:
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