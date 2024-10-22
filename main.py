import json
import os
import time

import openai
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_core.load import dumpd, dumps, load, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

load_dotenv()
os.environ["CURL_CA_BUNDLE"] = ""

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
EMBEDDING_MODEL_NAME = config["embedding_name"]
EMBEDDING_MODEL_TYPE = config["embedding_type"]
DATABASE_NAME = config["collection_name"]
LLM_MODEL_NAME = config["llm_name"]
LLM_MODEL_TYPE = config["llm_type"]
# get secrets
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


def invoke_rag(question):

    prompt = load_chat_prompt_template()
    # Prompt Engineering on the question for better similarity_search
    retrieved_docs = retriever.invoke(question)

    # Retrieve relevant chunks based on the question
    start_time = time.time()
    print(f" Time for creating the retriever: {time.time() - start_time}")
    retrieved_docs = retriever.invoke(input=question)
    print(f" Total time for retrieving documents: {time.time() - start_time}")

    # Prompt Engineering on the question for better question

    # Build chains
    start_time = time.time()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    # Invoke the retrieval chain
    result = retrieval_chain.invoke({"input": question})
    print(f" Total Time for all the chain: {time.time() - start_time}")

    return result


def create_retriever():
    if EMBEDDING_MODEL_TYPE == "AzureOpenAI":
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

    vectorstore = Chroma(
        collection_name=DATABASE_NAME,
        embedding_function=embedding_model,
        persist_directory=f"chroma_db/{EMBEDDING_MODEL_NAME}",
    )
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
        search_kwargs={"k": 3},
    )

    # Assuming the original text is stored under a "text" key in the metadata:
    doc_list = vectorstore.get()["documents"]
    metadata = vectorstore.get()["metadatas"]
    bm25_retriever = BM25Retriever.from_texts(doc_list, metadatas=metadata)
    bm25_retriever.k = 1
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, similarity_retriever], weights=[0.4, 0.6]
    )

    # Reranker
    # compressor = CohereRerank(
    #     top_n=3,
    # )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=ensemble_retriever
    # )
    return similarity_retriever


def get_llm_model():
    if LLM_MODEL_TYPE == "Local":
        llm = Ollama(model=LLM_MODEL_NAME)

        return llm
    if LLM_MODEL_TYPE == "HuggingFace":
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
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
        return llm

    if LLM_MODEL_TYPE == "AzureOpenAI":

        client = AzureChatOpenAI(
            azure_deployment=LLM_MODEL_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-06-01",
        )
        return client


def load_chat_prompt_template():
    # Define the prompt template using system and user messages
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a RAG solution for the user manual of the TC19, a textile card machine of Trützschler.
                Use the following pieces of context to answer the question or command at the end, only based on the context.
                If you cannot answer based on the context, just say that you don't know, don't try to make up an answer.
                \n\n{context}""",
            ),
            ("user", "{input}"),
        ]
    )
    return prompt


retriever = create_retriever()
llm = get_llm_model()

if __name__ == "__main__":
    question = "How can I optimise the carding-gap with T-CON and T-GO?"
    answer = invoke_rag(question)
    print(f"The Question:\n{question}")
    print(f"The Answer:\n{answer['answer']}\n")
    for i, con in enumerate(answer["context"]):
        print(f"Context {i}")
        print(f"Document: {answer['context'][0].metadata['source']}")
        print(f"Page: {answer['context'][0].metadata['page']}")
        print(f"Content: {answer['context'][0].page_content}\n")
