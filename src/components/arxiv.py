from typing import List

from langchain.schema import Document
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from src.log_utils import logger
from src.exception.exception import DocumentProcessingError


def arxiv_search(state) -> List[Document]:
    """
    Perform arxiv search based on the query.

    Args:
        state: State object containing the query and other parameters

    Returns:
        List of Document objects from arxiv search results

    Raises:
        DocumentProcessingError: If arxiv search fails
    """
    try:
        logger.info("Starting arxiv search")
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        docs = arxiv.run(state.query)
        logger.info(f"Found {len(docs)} documents from arxiv")
        return docs
    except Exception as e:
        raise DocumentProcessingError(
            "Failed to perform arxiv search",
            details={
                "query": state.query,
                "error": str(e)
            }
        )
