from typing import Optional
from pydantic import BaseModel, Field, confloat

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.log_utils import logger
from src.exception.exception import DocumentProcessingError


class ChunkRewriteResponse(BaseModel):
    """Response model for chunk rewriting with confidence scoring."""

    headline: str = Field(
        description="Generated headline that captures the main topic of the chunk"
    )


def rewrite_chunk(original_chunk: str, chapter_name: str) -> str:
    """
    Rewrites the chunk by generating a headline that captures its main topic.

    Args:
        original_chunk: The original chunk of text
        chapter_name: The chapter name

    Returns:
        The original chunk with a generated headline

    Raises:
        DocumentProcessingError: If chunk rewriting fails
    """
    try:
        logger.info("Starting chunk rewrite process")
        logger.debug("Original chunk preview: %s", original_chunk[:100])
        logger.debug("Chapter name: %s", chapter_name)

        # Input validation
        if not original_chunk or not chapter_name:
            raise DocumentProcessingError(
                "Invalid input for chunk rewriting",
                details={
                    "chapter_name": chapter_name,
                    "chunk_length": len(original_chunk) if original_chunk else 0,
                },
            )

        # Get the prompt from PromptManager
        try:
            prompt = PromptManager.get_prompt(
                "chunk_rewrite_prompt",
                original_chunk=original_chunk,
                chapter_name=chapter_name,
            )
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to generate chunk rewrite prompt",
                details={
                    "chapter_name": chapter_name,
                    "error": str(e),
                },
            )

        # Get LLM with structured output
        try:
            llm = LLMFactory(provider="azure")
            result = llm.create_completion(
                response_model=ChunkRewriteResponse,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise DocumentProcessingError(
                "LLM processing failed during chunk rewrite",
                details={
                    "chapter_name": chapter_name,
                    "error": str(e),
                },
            )

        logger.debug("Generated headline: %s", result.headline)
        logger.info("Chunk rewrite completed successfully")
        return f"{result.headline}---{original_chunk}"

    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            "Unexpected error during chunk rewriting",
            details={
                "chapter_name": chapter_name,
                "error": str(e),
            },
        )
