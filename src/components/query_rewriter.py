from pydantic import BaseModel, Field

from src.exception.exception import GenerationError
from src.log_utils import logger
from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager


class QueryRewriteResponse(BaseModel):
    rewritten_query: str = Field(description="The rewritten query optimized for search")


def rewrite_query(state: dict) -> dict:
    """
    Rewrites the query with a LLM to retrieve better results with similarity search.

    Args:
        state: The current graph state, contains the 'question'.
        rewrite_type: The type of query rewrite ("rewrite_query_arxiv" or other).

    Returns:
        dict: New key added to state, 'rewritten_query', containing the rewritten query.

    Raises:
        GenerationError: If query rewriting fails
    """
    rewrite_type = state.datasource
    try:
        # Validate input state
        if not hasattr(state, "question"):
            raise GenerationError(
                "Missing question in state for routing",
                details={"available_attributes": dir(state)},
            )

        # Retrieve the original query from the state
        original_query = state.question
        if not original_query.strip():
            raise GenerationError(
                "Empty question provided for query rewriting",
                details={"question_length": 0},
            )

        logger.info("Starting query rewriting for question: %s", original_query)

        # Get the prompt from PromptManager
        prompt = PromptManager.get_prompt(
            "query_rewrite_prompt", original_query=original_query
        )

        # Get LLM and rewrite query
        llm = LLMFactory(provider="azure")
        result = llm.create_completion(
            response_model=QueryRewriteResponse,
            messages=[{"role": "user", "content": prompt}],
        )

        logger.info("Query successfully rewritten")
        logger.debug("Rewritten query: %s", result.rewritten_query[:100])
        state.rewritten_query = result.rewritten_query

        return state

    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(
            "Failed to rewrite query",
            details={
                "question": state.question,
                "rewrite_type": rewrite_type,
                "error": str(e),
            },
        )
