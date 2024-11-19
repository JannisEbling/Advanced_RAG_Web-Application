import sys

from langchain_core.prompts import ChatPromptTemplate

from src.components.llms import get_llm_model
from src.config.prompts import (
    ALTERNATIVE_RESPONSE_PROMPT,
    RESPONSE_PROMPT,
)
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


def generate_response(state):
    """
    Generate a response from a LLM based on the question and documents in the state.

    Args:
        state (dict): The current graph state, containing 'question', 'documents', and 'hallucination_state'

    Returns:
        dict: New key added to state, 'response', which contains the generated LLM response
    """
    try:
        # Extract question and documents from the state
        question = state["question"]
        documents = state["documents"]
        hallucination_state = state.get("hallucination_state", False)

        # Log the details of the current question and documents
        logger.info("Generating response for the question: %s", question)
        logger.debug(
            "Documents to be used for context: %s",
            [doc.page_content[:100] for doc in documents],
        )  # Log the first 100 characters of each document for brevity

        # Determine the appropriate system prompt based on hallucination state
        system_prompt = (
            ALTERNATIVE_RESPONSE_PROMPT if hallucination_state else RESPONSE_PROMPT
        )

        # Create the prompt using the provided question and documents
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
            ]
        )
        document_texts = [doc.page_content for doc in documents]
        formatted_prompt = prompt_template.format_messages(
            input=question, context="\n".join(document_texts)
        )

        # Log the formatted prompt that will be sent to the LLM
        logger.debug("Formatted prompt to LLM: %s", formatted_prompt)

        # Get the LLM model and generate the response
        llm = get_llm_model("base_azure")
        response = llm(formatted_prompt)

        # Log the response received from the LLM
        logger.info(
            "Response generated from LLM: %s", response.content[:200]
        )  # Log the first 200 characters for brevity

        # Return the response as part of the state
        return {"response": response.content}

    except Exception as e:
        logger.error("Error occurred while generating LLM response", exc_info=True)
        # Raise a custom exception to handle this error further up the chain
        raise MultiAgentRAGException("Failed to generate LLM response", sys) from e
