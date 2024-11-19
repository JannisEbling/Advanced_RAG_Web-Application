import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pprint import pprint
from typing import List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from src.logging import logger, response_logger
from src.components.arxiv import arxiv_search
from src.components.detect_hallucination import (
    is_answer_grounded_on_context,
)
from src.components.generate_response import generate_response
from src.components.query_rewriter import rewrite_query
from src.components.reranker import rerank_documents
from src.components.retriever import retrieve
from src.components.route import route_question


def invoke_single(user_input):
    """
    Answers the users question with an appropiate response

    Args:
        user_input: The users question

    Returns:
        workflow output artifact that contains retrieved documents, generated answer and more
    """

    app = _initiate_workflow()
    logger.info("Start worklow!")
    # output = app.stream(user_input)
    output = app.invoke({"question": user_input})
    logger.info("Workflow finished!")
    response_logger.info(output)
    return output


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: query
        rewritten_query: rewritten query
        response: LLM generation
        documents: list of all retrieved documents
        reranked_documents: list of reranked documents
        hallucination_state: boolean of hallucination state

    """

    question: str
    rewritten_query: str
    response: str
    documents: List[str]
    reranked_documents: List[str]
    hallucination_state: bool


def _initiate_workflow():
    """
    Defines the workflow

    Returns:
        complete workflow
    """

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("arxiv_search", arxiv_search)  # search in arXiv database
    workflow.add_node("retrieve", retrieve)  # retrieve documents from vectorstore
    workflow.add_node(
        "rewrite_query_vectorstore",
        lambda state: rewrite_query(state, "rewrite_query_vectorstore"),
    )  # rephrase and optimize the query
    workflow.add_node(
        "rewrite_query_arxiv",
        lambda state: rewrite_query(state, "rewrite_query_arxiv"),
    )  # rephrase and optimize the query
    workflow.add_node(
        "generate_response", generate_response
    )  # create response based on retrieved data
    workflow.add_node("reranker", rerank_documents)  # rank documents for relevance
    workflow.add_node(
        "generate_alternative_response", generate_response
    )  # create alternative response due to hallucination

    ## Build the graph
    # Route to fitting agent
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "arxiv_search": "rewrite_query_arxiv",  # route to arXiv search
            "vectorstore": "rewrite_query_vectorstore",  # route to vectorstore search
        },
    )

    # vectorstore search
    workflow.add_edge("rewrite_query_vectorstore", "retrieve")
    workflow.add_edge("retrieve", "reranker")
    workflow.add_edge("reranker", "generate_response")

    # arXiv search
    workflow.add_edge("rewrite_query_arxiv", "arxiv_search")
    workflow.add_edge("arxiv_search", "generate_response")

    # Halucination handling
    workflow.add_conditional_edges(
        "generate_response",
        is_answer_grounded_on_context,
        {
            "grounded on context": END,  # finish if response is grounded
            "hallucination": "generate_alternative_response",  # handle hallucinations if detected
        },
    )
    workflow.add_edge(
        "generate_alternative_response", END
    )  # generate alternative response

    # Compile the workflow
    app = workflow.compile()
    return app


if __name__ == "__main__":
    response_logger.info("test")
    application = _initiate_workflow()
    inputs = {"question": "What is logistic regression?"}
    value = None
    for _output in application.stream(inputs):
        for key, value in _output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

        # Final generation
    response_logger.info(value)
    pprint(value)
    print("END")
