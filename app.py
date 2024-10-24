import requests
import streamlit as st

from main import invoke_rag


def get_response(user_input):
    return invoke_rag(user_input)


# Set the title of the app
st.title("LLM Question Answering System")

# Info Icon for additional information
st.sidebar.info(
    "This app allows you to ask questions about LLMs.\n\n"
    "Type your question in the box below and click 'Submit'.\n"
    "You will receive an answer from an advanced AI language model (RAG-based)."
)

# User input for question about LLM
user_input = st.text_input("Ask a question about LLMs:")


if st.button("Submit"):
    if user_input:
        answer = get_response(user_input)
        st.write(answer)

    else:
        st.write("Please enter a question.")
