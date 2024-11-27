import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from src.log_utils import logger, response_logger
from src.exception.exception import (
    GenerationError,
    RetrievalError,
    RoutingError,
    RAGPipelineError,
)
from src.components.arxiv import arxiv_search
from src.components.detect_hallucination import is_answer_grounded_on_context
from src.components.generate_response import ResponseGenerator
from src.components.query_rewriter import rewrite_query
from src.components.reranker import rerank_documents
from src.components.route import route_question, DataSource


class WorkflowState(BaseModel):
    """Represents the state of our RAG pipeline workflow."""

    # Input state
    question: str = Field(description="Original user question")

    # Routing state
    datasource: Optional[DataSource] = Field(
        None, description="Selected data source (arxiv or vectorstore)"
    )
    routing_confidence: Optional[float] = Field(
        None, description="Confidence in routing decision"
    )

    # Query processing state
    rewritten_query: Optional[str] = Field(
        None, description="Optimized version of the query"
    )
    query_confidence: Optional[float] = Field(
        None, description="Confidence in query rewrite"
    )

    # Retrieval state
    documents: List[str] = Field(
        default_factory=list, description="Retrieved documents"
    )
    reranked_documents: List[str] = Field(
        default_factory=list, description="Reranked documents"
    )
    reranking_confidence: Optional[float] = Field(
        None, description="Confidence in document ranking"
    )

    # Response state
    response: Optional[str] = Field(None, description="Generated response")
    is_hallucination: Optional[bool] = Field(
        None, description="Whether response contains hallucinations"
    )
    response_confidence: Optional[float] = Field(
        None, description="Confidence in response quality"
    )


def invoke(user_input: str) -> dict:
    """
    Process a user question through the RAG pipeline.

    Args:
        user_input: The user's question

    Returns:
        Complete workflow output including retrieved documents, response, and confidence scores

    Raises:
        RoutingError: If question routing fails
        RetrievalError: If document retrieval fails
        GenerationError: If response generation fails
    """
    workflow = _create_workflow()
    logger.info("Starting RAG pipeline workflow")

    try:
        output = workflow.invoke({"question": user_input})
        logger.info("Workflow completed successfully")
        response_logger.info("Generated response for query: %s", user_input)
        response_logger.info(
            "Response: %s", output.get("response", "No response generated")
        )
        return output
    except Exception as e:
        error_msg = f"RAG pipeline failed for query: {user_input}"
        logger.error(error_msg, exc_info=True)

        if "routing" in str(e).lower():
            raise RoutingError(
                "Failed to route question", details={"query": user_input}
            )
        elif "retrieval" in str(e).lower():
            raise RetrievalError(
                "Failed to retrieve relevant documents", details={"query": user_input}
            )
        else:
            raise GenerationError(
                "Failed to generate response", details={"query": user_input}
            )


def _create_workflow() -> StateGraph:
    """
    Creates the RAG pipeline workflow graph.

    Returns:
        Configured workflow graph
    """
    workflow = StateGraph(WorkflowState)

    # Core processing nodes
    workflow.add_node("route", route_question)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("arxiv_search", arxiv_search)
    workflow.add_node("vectorstore_search", retrieve)
    workflow.add_node("rerank", rerank_documents)
    workflow.add_node("generate", generate_response)
    workflow.add_node("detect_hallucination", is_answer_grounded_on_context)
    workflow.add_node("generate_fallback", generate_response)

    # Define workflow edges
    workflow.add_edge(START, "route")

    # Route based on data source
    workflow.add_conditional_edges(
        "route",
        lambda x: x["datasource"].value,
        {
            "arxiv": "arxiv_search",
            "vectorstore": "vectorstore_search",
        },
    )

    # Common processing path
    for source in ["arxiv_search", "vectorstore_search"]:
        workflow.add_edge(source, "rerank")

    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "detect_hallucination")

    # Handle hallucinations
    workflow.add_conditional_edges(
        "detect_hallucination",
        lambda x: "generate_fallback" if x["is_hallucination"] else END,
        {"generate_fallback": END, END: END},
    )

    return workflow


if __name__ == "__main__":
    # Example usage with error handling
    try:
        question = "What is backpropagation?"
        result = invoke(question)
        print(f"Response: {result.get('response', 'No response generated')}")
        print(f"Confidence: {result.get('response_confidence', 0):.2f}")
    except (RoutingError, RetrievalError, GenerationError) as e:
        logger.error("Pipeline error: %s", str(e))
