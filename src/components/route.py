from typing import Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, confloat

from src import logger, RoutingError
from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager


class DataSource(str, Enum):
    """Enum for data sources that can be used for answering questions."""

    ARXIV = "arxiv"
    VECTORSTORE = "vectorstore"


class RouteResponse(BaseModel):
    """Response model for question routing."""

    datasource: DataSource = Field(
        description="The data source to use for answering the question"
    )
    decision_reason: str = Field(
        description="Explanation for why this data source was chosen"
    )
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Confidence in the routing decision"
    )


def route_question(state: Dict[str, Any]) -> str:
    """
    Route question to arXiv search or vectorstore search based on content analysis.

    Args:
        state: The current graph state containing the question

    Returns:
        Next node to call based on routing decision

    Raises:
        RoutingError: If routing decision fails
    """
    try:
        # Validate state
        if "question" not in state:
            raise RoutingError(
                "Missing question in state for routing",
                details={"available_keys": list(state.keys())},
            )

        question = state["question"]
        if not question.strip():
            raise RoutingError(
                "Empty question provided for routing",
                details={"question": question},
            )

        logger.info("Starting routing process for question")
        logger.debug("Routing question: %s", question)

        try:
            # Get the prompt from PromptManager
            prompt = PromptManager.get_prompt("routing_prompt", question=question)
        except Exception as e:
            raise RoutingError(
                "Failed to generate routing prompt",
                details={
                    "question": question,
                    "error": str(e),
                },
            )

        try:
            # Get LLM and make routing decision
            llm = LLMFactory(provider="azure")
            result = llm.create_completion(
                response_model=RouteResponse,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise RoutingError(
                "LLM processing failed during routing",
                details={
                    "question": question,
                    "error": str(e),
                },
            )

        logger.debug("Decision reasoning: %s", result.decision_reason)
        logger.debug("Data source: %s", result.datasource)
        logger.debug("Confidence score: %.2f", result.confidence_score)

        # Store the routing decision in state for potential use downstream
        state["routing"] = {
            "decision_reason": result.decision_reason,
            "datasource": result.datasource,
            "confidence_score": result.confidence_score,
        }

        # Handle low confidence cases
        if result.confidence_score < 0.7:
            logger.warning(
                "Low confidence in routing decision (%.2f), defaulting to vectorstore",
                result.confidence_score,
            )
            return "rewrite_query_vectorstore"

        # Route based on decision
        if result.datasource == DataSource.ARXIV:
            logger.info(
                "Routing to arXiv search with confidence %.2f",
                result.confidence_score,
            )
            return "rewrite_query_arxiv"
        else:
            logger.info(
                "Routing to vectorstore search with confidence %.2f",
                result.confidence_score,
            )
            return "rewrite_query_vectorstore"

    except RoutingError:
        raise
    except Exception as e:
        raise RoutingError(
            "Unexpected error during question routing",
            details={
                "question": state.get("question", ""),
                "error": str(e),
            },
        )
