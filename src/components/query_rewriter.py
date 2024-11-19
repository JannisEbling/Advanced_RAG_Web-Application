from pydantic import BaseModel, Field

from src import logger, GenerationError
from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager


class QueryRewriteResponse(BaseModel):
    rewritten_query: str = Field(description="The rewritten query optimized for search")


def rewrite_query(state: dict, rewrite_type: str) -> dict:
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
    try:
        # Validate input state
        if "question" not in state:
            raise GenerationError(
                "Missing question in state for query rewriting",
                details={
                    "available_keys": list(state.keys()),
                    "required_keys": ["question"]
                }
            )

        # Retrieve the original query from the state
        original_query = state["question"]
        if not original_query.strip():
            raise GenerationError(
                "Empty question provided for query rewriting",
                details={"question_length": 0}
            )

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

    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(
            "Failed to rewrite query",
            details={
                "question": state.get("question", ""),
                "rewrite_type": rewrite_type,
                "error": str(e)
            }
        )
