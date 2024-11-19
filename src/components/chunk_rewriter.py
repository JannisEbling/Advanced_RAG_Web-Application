import sys
from pydantic import BaseModel, Field

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class ChunkRewriteResponse(BaseModel):
    rewritten_chunk: str = Field(description="The rewritten and optimized chunk of text")


def rewrite_chunk(original_chunk, chapter_name):
    """
    Rewrites the chunk with a LLM to retrieve better results with similarity search and add a headline with the chapter name

    Args:
        original_chunk (str): The original chunk
        chapter_name (str): The chapter name

    Returns:
        cleaned_text (str): The rewritten chunk for optimzed retrieval
    """
    try:
        logger.info("Starting chunk rewrite process.")
        logger.debug("Original chunk: %s", original_chunk[:100])
        logger.debug("Chapter name: %s", chapter_name)

        # Get the prompt from PromptManager
        prompt = PromptManager.get_prompt("chunk_rewrite_prompt", 
                                        original_chunk=original_chunk,
                                        chapter_name=chapter_name)

        # Get LLM with structured output
        llm = LLMFactory(provider="azure")
        response = llm.create_completion(
            response_model=ChunkRewriteResponse,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.debug("Rewritten chunk: %s", response.rewritten_chunk[:100])
        logger.info("Chunk rewrite completed successfully.")
        return response.rewritten_chunk

    except Exception as e:
        logger.error("An error occurred during chunk rewriting", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to rewrite chunk", sys
        ) from e
