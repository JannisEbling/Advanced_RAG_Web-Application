import os
import sys

import torch
import yaml
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from multi_agent_rag import constants
from multi_agent_rag.exception.exception import MultiAgentRAGException
from multi_agent_rag.logging import logger


def get_llm_model(llm_config_path: str = "base_azure"):
    """
    Loads an LLM (Large Language Model) based on the configuration file provided.

    Args:
        llm_config_path (str): Path to the LLM configuration file (YAML).

    Returns:
        llm: An LLM instance that can be queried.
    """
    try:
        logger.info("Loading LLM model from configuration at: %s", llm_config_path)

        if llm_config_path.endswith(".yaml"):
            llm_config_path = llm_config_path[:-5]

        # Load configuration from YAML file
        with open(
            f"{constants.LLM_CONFIG_DIR}/{llm_config_path}.yaml",
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)
        LLM_MODEL_NAME = config["LLM_MODEL_NAME"]
        LLM_MODEL_TYPE = config["LLM_MODEL_TYPE"]

        if LLM_MODEL_TYPE == "HuggingFace":
            logger.info("Loading HuggingFace model: %s", LLM_MODEL_NAME)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

            llm = pipeline(
                model=model,
                tokenizer=tokenizer,
                task=config.TASK,
                do_sample=True,
                temperature=config.TEMPERATURE,
                repetition_penalty=config.REPETITION_PENALTY,
                return_full_text=False,
                max_new_tokens=config.MAX_NEW_TOKENS,
            )
            logger.info("Successfully loaded HuggingFace model: %s", LLM_MODEL_NAME)
            return llm

        if LLM_MODEL_TYPE == "AzureOpenAI":
            logger.info("Loading Azure OpenAI model: %s", LLM_MODEL_NAME)

            AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
            AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

            client = AzureChatOpenAI(
                azure_deployment=LLM_MODEL_NAME,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=config["API_VERSION"],
            )
            logger.info("Successfully loaded Azure OpenAI model: %s", LLM_MODEL_NAME)
            return client

        if LLM_MODEL_TYPE == "Groq":
            logger.info("Loading Groq model: %s", LLM_MODEL_NAME)

            GROQ_API_KEY = os.getenv("GROQ_API_KEY")

            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL_NAME)
            logger.info("Successfully loaded Groq model: %s", LLM_MODEL_NAME)
            return llm

        else:
            logger.error("Invalid LLM model type: %s", LLM_MODEL_TYPE)
            raise ValueError(f"No valid LLM model type selected: {LLM_MODEL_TYPE}")

    except Exception as e:
        logger.error("Error occurred while loading LLM model", exc_info=True)
        raise MultiAgentRAGException("Failed to load LLM model", sys) from e
