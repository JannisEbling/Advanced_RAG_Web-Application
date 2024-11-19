import sys
from typing import Literal

from pydantic import BaseModel, Field

from src.components.llms import get_llm_model
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "arxiv_search"] = Field(
        ...,
        description="Given a user question choose to route it to arXiv or a vectorstore.",
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

        # Get LLM with structured output
        llm = get_llm_model("base_azure")
        structured_llm_router = llm.with_structured_output(RouteQuery)
        
        # Get routing decision
        source = structured_llm_router.invoke(prompt)
        logger.debug("Routing decision: %s", source.datasource)

        if source.datasource == "arxiv_search":
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
