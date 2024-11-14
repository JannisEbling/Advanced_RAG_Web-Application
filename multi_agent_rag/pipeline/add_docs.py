import os

import yaml
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_agent_rag import constants
from multi_agent_rag.components.chunk_rewriter import rewrite_chunk
from multi_agent_rag.components.doc_cleaner import (
    filter_recurrent_obsolescences_with_remove,
    replace_new_line_with_space,
    replace_t_with_space,
    split_into_chapters,
)
from multi_agent_rag.components.embedding import get_embedding_model

os.environ["CURL_CA_BUNDLE"] = ""
load_dotenv()

# get config information
with open(
    f"{constants.EMBEDDING_CONFIG_DIR}/base_azure.yaml",
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)
EMBEDDING_MODEL_NAME = config["embedding_name"]
EMBEDDING_MODEL_TYPE = config["embedding_type"]


def add_docs(doc_path: str):
    """
    Adds a document to the vectorstore

    Args:
        doc_path: Path to the document
    """

    # splits documents by chapters
    documents = split_into_chapters(doc_path)

    # create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, add_start_index=True, chunk_overlap=200
    )

    splitted_documents = []
    for doc in documents[1:2]:
        # Extract chapter metadata and content
        chapter_title = doc.metadata.get("chapter", "")
        chapter_text = f"{doc.page_content}"

        # Split the document text into chunks
        chunks = text_splitter.split_text(chapter_text)

        # Create new Document objects for each chunk, including chapter metadata
        for chunk in chunks[:5]:
            # Clean the chunk
            chunk_content = replace_new_line_with_space(chunk)
            chunk_content = filter_recurrent_obsolescences_with_remove(chunk_content)
            chunk_content = replace_t_with_space(chunk_content)
            original_chunk = chunk_content

            # Rewrite the chunk
            chunk_content = rewrite_chunk(chunk_content, chapter_title)

            split_doc = Document(
                page_content=chunk_content,
                metadata={"chapter": chapter_title, "original_chunk": original_chunk},
            )
            splitted_documents.append(split_doc)

    # create embedding model
    embedding_model = get_embedding_model("base_azure")

    # create vectorstore reference
    vectorstore = Chroma(
        collection_name=EMBEDDING_MODEL_NAME,
        embedding_function=embedding_model,
        persist_directory=constants.VECTOR_DATABASE_DIR,
    )

    # Add documents to the vectorstore
    vectorstore.add_documents(splitted_documents)


if __name__ == "__main__":
    add_docs("C:/Projekte/Data_Science/Advanced_RAG_Web-Application/data/Chapter4.pdf")
