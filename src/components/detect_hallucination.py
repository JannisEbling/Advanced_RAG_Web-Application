import sys

from pydantic import BaseModel, Field

from src.components.llms import get_llm_model
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class IsGroundedOnFacts(BaseModel):
    """
    Output schema for the grounded-on-facts detection.
    """

    grounded_on_facts: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def is_answer_grounded_on_context(state):
    """
    Determines if the answer is a hallucination or not. Routes to the corresponding node.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    try:
        logger.info("Starting hallucination detection process.")
        context = state["reranked_documents"]
        answer = state["response"]

        logger.debug("Context for grounding check: %s", context[:100])
        logger.debug("Answer to check for grounding: %s", answer[:100])

        # Get the prompt from PromptManager
        prompt = PromptManager.get_prompt("hallucination_detection_prompt", 
                                        context=context,
                                        answer=answer)

        # Get LLM with structured output
        llm = get_llm_model("base_azure")
        structured_llm = llm.with_structured_output(IsGroundedOnFacts)
        
        # Check if answer is grounded on facts
        result = structured_llm.invoke(prompt)
        logger.debug("Grounding check result: %s", result.grounded_on_facts)

        if result.grounded_on_facts:
            logger.info("Answer is grounded on facts")
            return "end"
        else:
            logger.info("Answer is not grounded on facts, regenerating response")
            return "generate_alternative_response"

    except Exception as e:
        logger.error("An error occurred during hallucination detection", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to check if answer is grounded on facts", sys
        ) from e
