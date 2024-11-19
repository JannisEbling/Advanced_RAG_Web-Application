import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from src import config as app_config
from src.components.embedding import get_embedding_model
from src.exception.exception import MultiAgentRAGException
from src.logging import logger

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = ""

with open(
    f"{app_config.EMBEDDING_CONFIG_DIR}/base_azure.yaml",
    "r",
    encoding="utf-8",
) as file:
    embedding_config = yaml.safe_load(file)
EMBEDDING_MODEL_NAME = embedding_config["embedding_name"]


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

        try:
            embedding_model = get_embedding_model("base_azure")
        except MultiAgentRAGException:
            # Re-raise MultiAgentRAGException as is
            raise

        vectorstore = Chroma(
            collection_name=EMBEDDING_MODEL_NAME,
            embedding_function=embedding_model,
            persist_directory=app_config.VECTOR_DATABASE_DIR,
        )
        logger.info("Vectorstore created successfully.")
        return vectorstore, embedding_model

    except Exception as e:
        logger.error("Failed to create vectorstore.", exc_info=True)
        try:
            raise ValueError("Vectorstore creation failed") from e
        except ValueError as ve:
            raise MultiAgentRAGException(str(ve), sys) from ve


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
    prompt = "Explain Linear Regression"
    vector_store, embeddings = create_vectorstore()

    # Debug: Print collection info
    print("\nVectorstore Info:")
    print(f"Collection name: {vector_store._collection.name}")
    print(f"Number of documents: {vector_store._collection.count()}")
    print(f"Collection metadata: {vector_store._collection.metadata}")
    print(f"Persist directory: {vector_store._persist_directory}")

    # Try listing all collections
    print("\nAll Collections:")
    client = vector_store._client
    collections = client.list_collections()
    for collection in collections:
        print(f"Collection: {collection.name}")
        print(f"Count: {collection.count()}")

    # Try getting embeddings directly from database
    print("\nTrying direct database access:")
    import sqlite3

    conn = sqlite3.connect(f"{vector_store._persist_directory}/chroma.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables:", cursor.fetchall())
    cursor.execute("SELECT COUNT(*) FROM embeddings;")
    print("Embeddings count:", cursor.fetchone()[0])
    conn.close()

    # Debug: Print collection info
    print("\nVectorstore Info:")
    print(f"Collection name: {vector_store._collection.name}")
    print(f"Number of documents: {vector_store._collection.count()}")
    print(f"Collection metadata: {vector_store._collection.metadata}")

    # Try retrieving documents
    docs = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=embeddings.embed_query(prompt), k=5
    )
    print("\nRetrieved Documents:")
    for doc in docs:
        print(f"\nDocument content: {doc[1][:200]}...")
        print(f"Document metadata: {doc[2]}")
    print(docs)
