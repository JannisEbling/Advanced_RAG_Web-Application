import sys
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llms import get_llm_model
from src.config.prompts import ROUTING_PROMPT
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

        llm = get_llm_model("base_azure")
        structured_llm_router = llm.with_structured_output(RouteQuery)

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTING_PROMPT),
                ("human", "{question}"),
            ]
        )
        question_router = route_prompt | structured_llm_router

        source = question_router.invoke({"question": question})
        logger.debug("Routing decision: %s", source.datasource)

        if source.datasource == "arxiv_search":
            logger.info("Routing question to arXiv search.")
            return "arxiv_search"
        elif source.datasource == "vectorstore":
            logger.info("Routing question to vectorstore.")
            return "vectorstore"
        else:
            logger.warning(
                "Routing decision returned an unexpected datasource: %s",
                source.datasource,
            )
            return "unknown_source"

    except Exception as e:
        logger.error("An error occurred while routing the question.", exc_info=True)
        raise MultiAgentRAGException("Failed to route question", sys) from e
