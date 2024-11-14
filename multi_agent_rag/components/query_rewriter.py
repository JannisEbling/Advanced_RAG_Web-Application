import sys

from langchain.prompts import PromptTemplate

from multi_agent_rag.components.llms import get_llm_model
from multi_agent_rag.constants.prompts import (
    ARXIV_QUERY_REWRITE_PROMPT,
    VECTORSTORE_QUERY_REWRITE_PROMPT,
)
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger


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
        if rewrite_type == "rewrite_query_arxiv":
            template = VECTORSTORE_QUERY_REWRITE_PROMPT
        else:
            template = ARXIV_QUERY_REWRITE_PROMPT

        # Create the prompt for query rewriting
        query_rewrite_prompt = PromptTemplate(
            input_variables=["original_query"], template=template
        )

        # Get the LLM model to rewrite the query
        llm = get_llm_model("base_azure")
        query_rewriter = query_rewrite_prompt | llm

        # Invoke the query rewriting process
        response = query_rewriter.invoke(original_query)

        logger.info("Query successfully rewritten.")
        return {"rewritten_query": response.content}

    except Exception as e:
        logger.error("An error occurred while rewriting the query.", exc_info=True)
        # Raise a custom exception with details
        raise MultiAgentRAGException("Query rewriting failed", sys) from e
