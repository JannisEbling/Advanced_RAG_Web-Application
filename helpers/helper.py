import os

from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["CURL_CA_BUNDLE"] = ""

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_db_save_path = "vector_database"

KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
    vector_db_save_path, embedding_model, allow_dangerous_deserialization=True
)


def invoke_rag(question):

    # Prompt Engineering on the question for better similarity_search

    # Retrieve relevant chunks based on the question
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=question, k=2)
    print(retrieved_docs)

    # ReRank Documents

    # Prompt Engineering on the question for better question

    # Load Prompt_template

    # Create final Prompt
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    final_prompt = f"Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer. Context: {context}. And the question is: {question}"

    # Redact an answer
    llm = Ollama(model="llama2")
    answer = "answer"  # llm(final_prompt)
    return answer, retrieved_docs_text


if __name__ == "__main__":
    answer, chunks = invoke_rag("What information does the type plate contain?")
    print(answer)
