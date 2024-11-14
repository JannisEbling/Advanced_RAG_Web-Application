import sys

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from multi_agent_rag.components.llms import get_llm_model
from multi_agent_rag.constants.prompts import RERANKING_PROMPT
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger


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

        prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template=RERANKING_PROMPT,
        )

        llm = get_llm_model("base_azure")
        llm_chain = prompt_template | llm.with_structured_output(RatingScore)

        scored_docs = []

        for doc in docs:
            input_data = {"query": query, "doc": doc.page_content}
            try:
                score = llm_chain.invoke(input_data).relevance_score
                score = float(score)
            except (ValueError, TypeError):
                logger.warning("Failed to parse score for document: %s", doc)
                score = 0

            scored_docs.append((doc, score))

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        logger.info("Top %d documents ranked successfully.", top_n)
        return {"reranked_documents": [doc for doc, _ in reranked_docs[:top_n]]}

    except Exception as e:
        logger.error("An error occurred while reranking documents.", exc_info=True)
        raise MultiAgentRAGException("Document reranking failed", sys) from e
