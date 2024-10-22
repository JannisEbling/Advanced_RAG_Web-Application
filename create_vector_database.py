import os

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["CURL_CA_BUNDLE"] = ""
load_dotenv()

# get config information
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
EMBEDDING_MODEL_NAME = config["embedding_name"]
EMBEDDING_MODEL_TYPE = config["embedding_type"]
DATABASE_NAME = config["collection_name"]


def create_update_vectordatabase():
    # load documents
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    # create and apply text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, add_start_index=True, chunk_overlap=50
    )
    splitted_documents = text_splitter.split_documents(documents)

    # create embedding model
    embedding_model = get_embedding_model()

    # create new vector database
    vectorstore = Chroma(
        collection_name=DATABASE_NAME,
        embedding_function=embedding_model,
        persist_directory=f"chroma_db/{EMBEDDING_MODEL_NAME}",
    )

    # Add documents to vector database
    vectorstore.add_documents(splitted_documents)


def get_embedding_model():
    if EMBEDDING_MODEL_TYPE == "AzureOpenAI":
        # get Azure OpenAI secrets
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

        embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=EMBEDDING_MODEL_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-06-01",
            dimensions=256,
        )

    if EMBEDDING_MODEL_TYPE == "HuggingFace":
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if EMBEDDING_MODEL_TYPE == "Local":
        assert EMBEDDING_MODEL_TYPE == "Local" "Local is not implemented yet"

    return embedding_model


if __name__ == "__main__":
    create_update_vectordatabase()
