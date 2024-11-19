import sys
from pydantic import BaseModel, Field

from src.components.llm_factory import LLMFactory
from src.prompts.prompt_manager import PromptManager
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class RatingScore(BaseModel):
    relevance_score: float = Field(
        ..., description="The relevance score of a document to a query."
    )


def rerank_documents(state, top_n: int = 3):
    """
    Reranks documents based on relevance to a query using a LLM.

    Args:
        state (dict): Current graph state, containing the 'question' and 'documents'.
        top_n (int): Number of top documents to return based on relevance.

    Returns:
        dict: Updated state with the 'reranked_documents' key, containing the top_n reranked documents.
    """
    try:
        query = state["question"]
        docs = state["documents"]

        logger.info("Reranking documents for query: %s", query)
        llm = LLMFactory(provider="azure")

        scored_docs = []
        for doc in docs:
            try:
                # Get prompt from PromptManager for each document
                prompt = PromptManager.get_prompt("reranking_prompt", 
                                                query=query,
                                                doc=doc.page_content)
                
                # Get relevance score using LLM
                result = llm.create_completion(
                    response_model=RatingScore,
                    messages=[{"role": "user", "content": prompt}]
                )
                score = float(result.relevance_score)
                scored_docs.append((doc, score))
                logger.debug("Document scored with relevance: %f", score)
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse score for document: %s", doc)
                continue

        # Sort documents by score in descending order and take top_n
        reranked_docs = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_n]]
        
        logger.info("Successfully reranked documents")
        logger.debug("Top %d documents selected", len(reranked_docs))

        return {"reranked_documents": reranked_docs}

    except Exception as e:
        logger.error("An error occurred while reranking documents", exc_info=True)
        raise MultiAgentRAGException(
            "Failed to rerank documents", sys
        ) from e
