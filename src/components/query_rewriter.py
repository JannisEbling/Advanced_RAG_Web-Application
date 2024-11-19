import sys
from pydantic import BaseModel, Field

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class QueryRewriteResponse(BaseModel):
    rewritten_query: str = Field(description="The rewritten query optimized for search")


def rewrite_query(state, rewrite_type):
    """
    Rewrites the query with a LLM to retrieve better results with similarity search.

    Args:
        state (dict): The current graph state, contains the 'question'.
        rewrite_type (str): The type of query rewrite ("rewrite_query_arxiv" or other).

    Returns:
        dict: New key added to state, 'rewritten_query', containing the rewritten query.
    """
    try:
        # Retrieve the original query from the state
        original_query = state["question"]
        logger.info("Starting query rewriting for question: %s", original_query)

        # Determine which template to use based on the rewrite_type
        template_name = "vectorstore_query_rewrite_prompt" if rewrite_type == "rewrite_query_arxiv" else "arxiv_query_rewrite_prompt"
        
        # Get the prompt from PromptManager
        prompt = PromptManager.get_prompt(template_name, original_query=original_query)
        
        # Get LLM and rewrite query
        llm = LLMFactory(provider="azure")
        result = llm.create_completion(
            response_model=QueryRewriteResponse,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.info("Query successfully rewritten")
        logger.debug("Rewritten query: %s", result.rewritten_query[:100])

        return {"rewritten_query": result.rewritten_query}

    except Exception as e:
        logger.error("An error occurred while rewriting query", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to rewrite query", sys
        ) from e
