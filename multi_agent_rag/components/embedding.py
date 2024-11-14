import os
import sys

import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from multi_agent_rag import constants
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger

os.environ["CURL_CA_BUNDLE"] = ""
load_dotenv()


def get_embedding_model(embedding_config_path):
    """
    Loads an embedding model based on the configuration file.

    Args:
        embedding_config_path (str): Path to the embedding config file (without the file extension).

    Returns:
        embedding: An embedding instance that can be queried, or None if no valid model type is found.
    """
    try:
        # Log that the process of loading the embedding configuration is starting
        logger.info(
            "Loading embedding model configuration from path: %s", embedding_config_path
        )
        if embedding_config_path.endswith(".yaml"):
            embedding_config_path = embedding_config_path[:-5]

        # Load configuration from YAML file
        with open(
            f"{constants.EMBEDDING_CONFIG_DIR}/{embedding_config_path}.yaml",
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        # Extract embedding model details from config
        EMBEDDING_MODEL_NAME = config["embedding_name"]
        EMBEDDING_MODEL_TYPE = config["embedding_type"]

        logger.debug("Embedding model name: %s", EMBEDDING_MODEL_NAME)
        logger.debug("Embedding model type: %s", EMBEDDING_MODEL_TYPE)

        # Check the embedding model type and initialize accordingly
        if EMBEDDING_MODEL_TYPE == "AzureOpenAI":
            AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
            AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

            # Log the use of AzureOpenAI embedding model
            logger.info("Using Azure OpenAI Embeddings model")

            embedding_model = AzureOpenAIEmbeddings(
                azure_deployment=EMBEDDING_MODEL_NAME,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2024-06-01",
                dimensions=256,
            )

        elif EMBEDDING_MODEL_TYPE == "HuggingFace":
            # Log the use of HuggingFace embedding model
            logger.info("Using HuggingFace Embeddings model")

            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        else:
            logger.error("Invalid embedding model type: %s", EMBEDDING_MODEL_TYPE)
            embedding_model = None

        if embedding_model:
            logger.info("Successfully loaded embedding model: %s", EMBEDDING_MODEL_NAME)
        else:
            logger.warning("No valid embedding model could be loaded.")

        return embedding_model

    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", e)
        raise MultiAgentRAGException(
            f"Configuration file not found: {embedding_config_path}.yaml", sys
        ) from e
    except KeyError as e:
        logger.error("Missing key in configuration file: %s", e)
        raise MultiAgentRAGException(
            f"Missing key in configuration file: {e}", sys
        ) from e
    except Exception as e:
        logger.error("Error occurred while loading the embedding model", exc_info=True)
        raise MultiAgentRAGException("Failed to load embedding model", sys) from e
