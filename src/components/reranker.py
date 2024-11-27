from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field, confloat

from langchain.docstore.document import Document
from src.log_utils import logger
from src.exception.exception import RetrievalError
from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager


class RerankingResponse(BaseModel):
    """Model for reranking response."""

    decision_reason: str = Field(description="Explanation of document relevance")
    relevance_score: confloat(ge=0.0, le=1.0) = Field(
        description="Relevance score for the document"
    )


def rerank_documents(state: Any, top_n: int = 3) -> Dict[str, List[Document]]:
    """
    Rerank documents based on their relevance to the query.

    Args:
        state: The current workflow state containing question and documents
        top_n: Number of top documents to return after reranking (default: 3)

    Returns:
        Updated state with reranked documents

    Raises:
        RetrievalError: If document reranking fails
    """
    try:
        if not hasattr(state, "question") or not hasattr(state, "documents"):
            raise RetrievalError(
                "Missing question in state for routing",
                details={"available_attributes": dir(state)},
            )

        query = state.question
        docs = state.documents

        if not docs:
            raise RetrievalError(
                "No documents provided for reranking",
                details={"query": query},
            )

        logger.info("Starting document reranking for %d documents", len(docs))

        # Initialize LLM for reranking
        llm = LLMFactory(provider="azure")

        # Store documents with their scores and explanations
        scored_docs: List[Tuple[Document, float, str]] = []

        # Process each document
        for doc in docs:
            try:
                # Get prompt from PromptManager for each document
                prompt = PromptManager.get_prompt(
                    "reranking_prompt",
                    query=query,
                    document=doc.page_content,
                )

                # Get structured reranking response using LLM
                result = llm.create_completion(
                    response_model=RerankingResponse,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Store document with its scores
                scored_docs.append(
                    (doc, result.relevance_score, result.decision_reason)
                )

                logger.debug(
                    "Document scored - Relevance: %.2f, Reason: %s",
                    result.relevance_score,
                    result.decision_reason[:100],
                )

            except Exception as e:
                logger.warning("Failed to score document, skipping. Error: %s", str(e))
                continue

        if not scored_docs:
            raise RetrievalError(
                "Failed to score any documents",
                details={
                    "query": query,
                    "num_input_docs": len(docs),
                },
            )

        # Sort documents by relevance score and take top_n
        reranked_docs = sorted(
            scored_docs,
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        # Add reranking explanations to document metadata
        final_docs = []
        for doc, score, reason in reranked_docs:
            doc.metadata["relevance_score"] = score
            doc.metadata["reranking_reason"] = reason
            final_docs.append(doc)

        logger.info("Successfully reranked documents")
        logger.debug(
            "Top %d documents selected from %d total", len(final_docs), len(scored_docs)
        )

        state.reranked_documents = final_docs
        return state

    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(
            "Failed to rerank documents",
            details={
                "error": str(e),
            },
        )
