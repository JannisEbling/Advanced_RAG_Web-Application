import sys
from typing import Literal
from pydantic import BaseModel, Field

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class RouteQuery(BaseModel):
    """Response model for query routing."""
    datasource: Literal["arxiv_search", "vectorstore_search"] = Field(
        description="The datasource to use for the query"
    )


def route_question(state):
    """
    Route question to arXiv search or vectorstore search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    try:
        logger.info("Starting routing process for the question.")
        question = state["question"]
        logger.debug("Routing question: %s", question)

        # Get the prompt from PromptManager
        prompt = PromptManager.get_prompt("routing_prompt", question=question)

        # Get LLM and make routing decision
        llm = LLMFactory(provider="azure")
        result = llm.create_completion(
            response_model=RouteQuery,
            messages=[{"role": "user", "content": prompt}]
        )
        
        logger.debug("Routing decision: %s", result.datasource)

        if result.datasource == "arxiv_search":
            logger.info("Routing to arXiv search")
            return "rewrite_query_arxiv"
        else:
            logger.info("Routing to vectorstore search")
            return "rewrite_query_vectorstore"

    except Exception as e:
        logger.error("An error occurred while routing question", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to route question", sys
        ) from e
