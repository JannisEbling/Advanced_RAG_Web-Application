from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain.docstore.document import Document
from src.log_utils import logger
from src.exception.exception import GenerationError
from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager


class GenerationResponse(BaseModel):
    """Model for generation response."""

    response: str = Field(description="Generated response text")
    figures: List[str] = Field(
        description="List of paths to figures that got referenced in the response"
    )
    confidence_score: float = Field(
        description="Confidence score for the generated response (0-1)",
        ge=0.0,
        le=1.0
    )


class ResponseGenerator:
    """Generates responses using LLM based on query and context documents."""

    def __init__(self, provider: str = "azure"):
        """
        Initialize ResponseGenerator with specified LLM provider.

        Args:
            provider: LLM provider to use (default: "azure")
        """
        self.provider = provider
        try:
            self.llm = LLMFactory(provider=provider)
        except Exception as e:
            raise GenerationError(
                "Failed to initialize ResponseGenerator",
                details={"provider": provider, "error": str(e)},
            )

    def generate_response(self, state: Any) -> Any:
        """
        Generate a response based on the query and reranked documents from state.

        Args:
            state: Current workflow state containing question and reranked documents

        Returns:
            Updated state with generated response

        Raises:
            GenerationError: If response generation fails
        """
        try:
            if not hasattr(state, "question") or not hasattr(
                state, "reranked_documents"
            ):
                raise GenerationError(
                    "Missing required state components for generation",
                    details={"available_attributes": dir(state)},
                )

            query = state.question
            context_docs = state.reranked_documents

            if not context_docs:
                raise GenerationError(
                    "No context documents provided for generation",
                    details={"query": query},
                )

            # Format context from reranked documents
            context_texts = []
            for doc in context_docs:
                # Include relevance info in context
                context = f"[Relevance: {doc.metadata.get('relevance_score', 'N/A')}]\n{doc.page_content}"
                context_texts.append(context)
            figure_captions = [figure["caption"] for figure in state.figures]
            figure_paths = [figure["path"] for figure in state.figures]
            formulas = [
                formula["reference"] + ": " + formula["metadata"]
                for formula in state.formulas
            ]

            # Get generation prompt
            prompt = PromptManager.get_prompt(
                "generation_prompt",
                query=query,
                context="\n\n".join(context_texts),
                figure_captions=figure_captions,
                figure_paths=figure_paths,
                formulas=formulas,
            )

            # Generate structured response
            result = self.llm.create_completion(
                response_model=GenerationResponse,
                messages=[{"role": "user", "content": prompt}],
            )

            logger.info("Successfully generated response")
            logger.debug("Response length: %d chars", len(result.response))
            logger.debug("Confidence score: %.2f", result.confidence_score)

            # Update state with response and sources
            state.response = result.response
            state.figure_paths = result.figures
            state.response_confidence = result.confidence_score
            return state

        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(
                "Failed to generate response",
                details={
                    "error": str(e),
                },
            )
