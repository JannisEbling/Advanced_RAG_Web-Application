import sys

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.components.llms import get_llm_model
from src.config.prompts import HALLUCINATION_DETECION_PROMPT
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

        llm = get_llm_model("base_azure")
        is_grounded_on_facts_prompt = PromptTemplate(
            template=HALLUCINATION_DETECION_PROMPT,
            input_variables=["context", "answer"],
        )
        is_grounded_on_facts_chain = (
            is_grounded_on_facts_prompt | llm.with_structured_output(IsGroundedOnFacts)
        )

        result = is_grounded_on_facts_chain.invoke(
            {"context": context, "answer": answer}
        )
        grounded_on_facts = result.grounded_on_facts

        if grounded_on_facts:
            logger.info("The answer is grounded in the facts.")
            return "grounded on context"
        else:
            logger.warning("The answer is identified as a hallucination.")
            return "hallucination"

    except Exception as e:
        logger.error("An error occurred during hallucination detection.", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to determine if the answer is grounded on context", sys
        ) from e
