import sys

from pydantic import BaseModel, Field

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class IsGroundedOnFacts(BaseModel):
    """Response model for hallucination detection."""
    grounded_on_facts: bool = Field(
        description="Whether the answer is grounded on facts from the context"
    )
    explanation: str = Field(
        description="Explanation of why the answer is or is not grounded on facts"
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
        llm = LLMFactory(provider="azure")
        result = llm.create_completion(
            response_model=IsGroundedOnFacts,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.debug("Grounding check result: %s", result.grounded_on_facts)
        logger.debug("Explanation: %s", result.explanation)

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
