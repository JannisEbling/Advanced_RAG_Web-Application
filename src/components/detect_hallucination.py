import sys
from enum import Enum
from pydantic import BaseModel, Field, confloat
from typing import List, Dict, Any

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger
from src.components.detect_hallucination import HallucinationDetector


class HallucinationDetectionResponse(BaseModel):
    """Response model for hallucination detection with confidence scoring."""
    decision_reason: str = Field(
        description="Detailed explanation of why the model thinks this is a hallucination or not, including specific examples and evidence"
    )
    is_hallucination: bool = Field(
        description="True if the answer contains hallucinated information, False if it is fully grounded in the context"
    )
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Confidence score between 0 and 1 indicating how certain the model is about its decision"
    )


def is_answer_grounded_on_context(state):
    """
    Determines if the answer is a hallucination or not using confidence scoring and detailed reasoning.

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
            response_model=HallucinationDetectionResponse,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.debug("Decision reasoning: %s", result.decision_reason)
        logger.debug("Is hallucination: %s", result.is_hallucination)
        logger.debug("Confidence score: %.2f", result.confidence_score)

        # Store the hallucination detection results in state for potential use downstream
        state["hallucination_detection"] = {
            "decision_reason": result.decision_reason,
            "is_hallucination": result.is_hallucination,
            "confidence_score": result.confidence_score
        }

        if not result.is_hallucination and result.confidence_score >= 0.7:
            logger.info("Answer is grounded on facts with high confidence")
            return "end"
        else:
            logger.info("Answer may be hallucinated or confidence is low, regenerating response")
            return "generate_alternative_response"

    except Exception as e:
        logger.error("An error occurred during hallucination detection", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to check if answer is grounded on facts", sys
        ) from e
