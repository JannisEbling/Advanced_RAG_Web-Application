import sys

from langchain.prompts import PromptTemplate

from multi_agent_rag.components.llms import get_llm_model
from multi_agent_rag.constants.prompts import CHUNK_REWRITE_PROMPT
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger


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

        chunk_rewrite_prompt = PromptTemplate(
            input_variables=["original_chunk", "chapter_name"],
            template=CHUNK_REWRITE_PROMPT,
        )
        llm = get_llm_model("base_azure")
        chunk_rewriter = chunk_rewrite_prompt | llm

        response = chunk_rewriter.invoke(
            {"original_chunk": original_chunk, "chapter_name": chapter_name}
        )

        logger.info("Chunk rewrite completed successfully.")
        return response.content

    except Exception as e:
        logger.error("An error occurred during chunk rewriting.", exc_info=True)
        raise MultiAgentRAGException("Failed to rewrite chunk", sys) from e
