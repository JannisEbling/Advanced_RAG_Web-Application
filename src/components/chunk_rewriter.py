import sys

from src.components.llms import get_llm_model
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


def rewrite_chunk(original_chunk, chapter_name):
    """
    Rewrites the chunk with a LLM to retrieve better results with similarity search and add a headline ith the chapter name

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
        # Get the prompt template directly from PromptManager
        prompt_template = PromptManager.get_prompt("chunk_rewrite_prompt", 
                                                 original_chunk=original_chunk,
                                                 chapter_name=chapter_name)
        
        llm = get_llm_model("base_azure")
        response = llm.invoke(prompt_template)

        logger.debug("Rewritten chunk: %s", response[:100])
        logger.info("Chunk rewrite completed successfully.")
        return response

    except Exception as e:
        logger.error("An error occurred during chunk rewriting.", exc_info=True)
        raise MultiAgentRAGException("Failed to rewrite chunk", sys) from e
