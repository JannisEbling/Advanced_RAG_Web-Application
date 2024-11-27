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


def write_header(doc) -> str:
    """
    Rewrites the chunk by generating a headline that captures its main topic.

    Args:
        doc (Document): The chunk to be rewritten

    Returns:
        The original chunk with a generated headline

    Raises:
        DocumentProcessingError: If chunk rewriting fails
    """
    try:
        # Get the prompt from PromptManager
        try:
            metadata = doc.metadata
            headline = (
                metadata.get("page_header")
                + " "
                + metadata.get("section_heading")
                + " "
                + metadata.get("subsection_heading")
            )
            prompt = PromptManager.get_prompt(
                "write_header_prompt",
                chunk=doc.page_content,
                headline=headline,
            )
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to generate chunk rewrite prompt",
                details={
                    "headline": headline,
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
                    "headline": headline,
                    "error": str(e),
                },
            )

        logger.debug("Generated headline: %s", result.headline)
        logger.info("Chunk rewrite completed successfully")
        return f"{result.headline}---{doc.page_content}"

    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            "Unexpected error during chunk rewriting",
            details={
                "headline": headline,
                "error": str(e),
            },
        )
