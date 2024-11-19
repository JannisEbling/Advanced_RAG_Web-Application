import sys

from src.components.llms import get_llm_model
from src.prompts.prompt_manager import PromptManager
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
        )

        # Get document texts
        document_texts = [doc.page_content for doc in documents]
        context = "\n".join(document_texts)

        # Get the appropriate prompt template based on hallucination state
        template_name = "alternative_response_prompt" if hallucination_state else "response_prompt"
        prompt = PromptManager.get_prompt(template_name, 
                                        input=question,
                                        context=context)

        # Generate response using the LLM
        llm = get_llm_model("base_azure")
        response = llm.invoke(prompt)

        logger.info("Response generated successfully")
        logger.debug("Generated response: %s", response[:100])  # Log first 100 chars

        # Update the state with the response
        state["response"] = response
        return state

    except Exception as e:
        logger.error("An error occurred while generating response", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to generate response", sys
        ) from e
