from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field, confloat
from langchain.schema import Document
from langchain_openai import AzureOpenAI

from src import logger, RetrievalError
from src.secure.secrets import secrets
from src.prompts.prompt_manager import PromptManager
from src.components.llm_factory import LLMFactory


class RerankingResponse(BaseModel):
    """Response model for document reranking with confidence scoring."""

    decision_reason: str = Field(
        description="Detailed explanation of why this document is relevant or not to the query, including specific matches and evidence"
    )
    relevance_score: confloat(ge=0.0, le=1.0) = Field(
        description="Relevance score between 0 and 1 indicating how relevant the document is to the query"
    )


def rerank_documents(state: Dict[str, Any], top_n: int = 3) -> Dict[str, List[Document]]:
    """
    Reranks documents based on relevance to a query using a LLM with structured output.

    Args:
        state: Current graph state, containing the 'question' and 'documents'
        top_n: Number of top documents to return based on relevance

    Returns:
        Updated state with the 'reranked_documents' key, containing the top_n reranked documents

    Raises:
        RetrievalError: If document reranking fails
    """
    try:
        if "question" not in state or "documents" not in state:
            raise RetrievalError(
                "Missing required state components for reranking",
                details={
                    "available_keys": list(state.keys()),
                    "required_keys": ["question", "documents"],
                },
            )

        query = state["question"]
        docs = state["documents"]

        if not docs:
            raise RetrievalError(
                "No documents provided for reranking",
                details={"query": query},
            )

        logger.info("Reranking documents for query: %s", query)

        try:
            llm = LLMFactory(provider="azure")
        except Exception as e:
            raise RetrievalError(
                "Failed to initialize LLM for reranking",
                details={"error": str(e)},
            )

        scored_docs: List[Tuple[Document, float, Dict[str, Any]]] = []
        for doc in docs:
            try:
                # Get prompt from PromptManager for each document
                prompt = PromptManager.get_prompt(
                    "reranking_prompt",
                    query=query,
                    doc=doc.page_content,
                )

                # Get structured reranking response using LLM
                result = llm.create_completion(
                    response_model=RerankingResponse,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Only consider documents with high confidence scores
                if result.confidence_score >= 0.7:
                    scored_docs.append(
                        (
                            doc,
                            result.relevance_score,
                            {
                                "decision_reason": result.decision_reason,
                                "relevance_score": result.relevance_score,
                            },
                        )
                    )
                    logger.debug(
                        "Document scored - Relevance: %.2f, Confidence: %.2f",
                        result.relevance_score,
                        result.confidence_score,
                    )
                else:
                    logger.debug(
                        "Document skipped due to low confidence: %.2f",
                        result.confidence_score,
                    )

            except Exception as e:
                logger.warning("Failed to process document: %s", str(e))
                continue

        if not scored_docs:
            raise RetrievalError(
                "No documents met the confidence threshold for reranking",
                details={
                    "query": query,
                    "total_docs": len(docs),
                },
            )

        # Sort documents by relevance score in descending order and take top_n
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_n]

        # Store reranking metadata in state
        state["reranking_metadata"] = {
            idx: metadata for idx, (_, _, metadata) in enumerate(reranked_docs)
        }

        # Extract just the documents for the final result
        final_docs = [doc for doc, _, _ in reranked_docs]

        logger.info("Successfully reranked documents")
        logger.debug("Top %d documents selected", len(final_docs))

        return {"reranked_documents": final_docs}

    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(
            "Unexpected error during document reranking",
            details={
                "query": state.get("question", ""),
                "error": str(e),
            },
        )
