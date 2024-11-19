import sys

from langchain.schema import Document
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from src.exception.exception import MultiAgentRAGException
from src.logging import logger


def arxiv_search(state):
    """
    arxiv search based on the query.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    try:
        logger.info("Starting arxiv search.")

        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        question = state["question"]

        logger.debug("Querying Arxiv with question: %s", question)
        docs = arxiv.invoke({"query": question})

        arxiv_results = Document(page_content=docs)
        logger.info("Arxiv search completed successfully.")

        return {"documents": arxiv_results, "query": question}

    except Exception as e:
        logger.error("An error occurred during Arxiv search.", exc_info=True)
        raise MultiAgentRAGException("Failed to complete Arxiv search", sys) from e
