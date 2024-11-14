import os
import sys

import yaml
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from multi_agent_rag import constants
from multi_agent_rag.components.embedding import get_embedding_model
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = ""

with open(
    f"{constants.EMBEDDING_CONFIG_DIR}/base_azure.yaml",
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)
EMBEDDING_MODEL_NAME = config["embedding_name"]


def create_vectorstore():
    """
    Create a vector store based on the specified embedding model.

    Returns:
        vectorstore: The created vectorstore.
        embedding_model: The embedding model used for creating the vectorstore.

    Raises:
        MultiAgentRAGException: If an invalid embedding model type is selected.
    """
    try:
        logger.info("Creating vectorstore with embedding model")

        embedding_model = get_embedding_model("base_azure")

        vectorstore = Chroma(
            collection_name=EMBEDDING_MODEL_NAME,
            embedding_function=embedding_model,
            persist_directory=constants.VECTOR_DATABASE_DIR,
        )
        logger.info("Vectorstore created successfully.")
        return vectorstore, embedding_model

    except Exception as e:
        logger.error("Failed to create vectorstore.", exc_info=True)
        raise MultiAgentRAGException("Vectorstore creation failed", sys) from e


def create_similarity_retriever(vectorstore=None):
    """
    Create a retriever using the vectorstore.

    Returns:
        similarity_retriever: The similarity-based retriever for retrieving documents.
    """
    try:
        logger.info("Creating retriever from vectorstore.")
        if not vectorstore:
            vectorstore, _ = create_vectorstore()

        # NOTE: Parent-Child Retriever is superior to normal retriever but at this time it can only use InMemoryStore
        # Possibly in the future there will be support for other storage methods or workaround methods will be available

        # store = InMemoryStore()
        # parent_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, add_start_index=True
        # )
        # child_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=400, add_start_index=True
        # )
        # retriever = ParentDocumentRetriever(
        #     vectorstore=vectorstore,
        #     docstore=store,
        #     child_splitter=child_splitter,
        #     parent_splitter=parent_splitter,
        #     search_kwargs={"k": 5},
        # )

        similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        logger.info("Retriever created successfully.")
        return similarity_retriever

    except Exception as e:
        logger.error("Failed to create retriever.", exc_info=True)
        raise MultiAgentRAGException("Retriever creation failed", sys) from e


def create_bm25_retriever(vectorstore=None):
    """
    Create a retriever using the vectorstore.

    Returns:
        similarity_retriever: The similarity-based retriever for retrieving documents.
    """
    try:
        logger.info("Creating retriever from vectorstore.")
        if not vectorstore:
            vectorstore, _ = create_vectorstore()
        doc_list = vectorstore.get()["documents"]
        metadata = vectorstore.get()["metadatas"]
        bm25_retriever = BM25Retriever.from_texts(doc_list, metadatas=metadata)
        bm25_retriever.k = 5
        logger.info("Retriever created successfully.")
        return bm25_retriever

    except Exception as e:
        logger.error("Failed to create retriever.", exc_info=True)
        raise MultiAgentRAGException("Retriever creation failed", sys) from e


def create_ensemble_retriever(vectorstore=None):
    try:
        logger.info("Creating retriever from vectorstore.")
        if not vectorstore:
            vectorstore, _ = create_vectorstore()
        bm25_retriever = create_bm25_retriever(vectorstore)
        similarity_retriever = create_similarity_retriever(vectorstore)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, similarity_retriever], weights=[0.4, 0.6]
        )
        return ensemble_retriever
    except Exception as e:
        logger.error("Failed to create retriever.", exc_info=True)
        raise MultiAgentRAGException("Retriever creation failed", sys) from e


try:
    retriever = create_ensemble_retriever()
except MultiAgentRAGException as e:
    logger.critical("Critical error initializing retriever: %s", e)
    raise


def retrieve(state):
    """
    Retrieve documents based on the question in the state.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with the 'documents' key containing retrieved documents.
    """
    try:
        question = state["question"]
        logger.info("Retrieving documents for question: %s", question)
        documents = retriever.invoke(question)
        logger.info("Documents retrieved successfully.")
        return {"documents": documents}

    except Exception as e:
        logger.error("Failed to retrieve documents.", exc_info=True)
        raise MultiAgentRAGException("Document retrieval failed", sys) from e


if __name__ == "__main__":
    prompt = "Explain Linear Regrssion"
    vector_store, embeddings = create_vectorstore()
    docs = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=embeddings.embed_query(prompt), k=5
    )
    for doc in docs:
        print(doc[1])
    print(docs)
