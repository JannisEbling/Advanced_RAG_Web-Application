import requests
import streamlit as st

from main import invoke_rag


def get_response(user_input):
    return invoke_rag(user_input)


# Set the title of the app
st.title("Machine TC-19 Question Answering System")

# Info Icon for additional information
st.sidebar.info(
    "This app allows you to ask questions about the Machine TC-19.\n\n"
    "Type your question in the box below and click 'Submit'.\n"
    "You will receive an answer from an advanced AI language model (RAG-based)."
)

# User input for question about Machine TC-19
user_input = st.text_input("Ask a question about Machine TC-19:")

# Add a button to submit the query
if st.button("Submit"):
    if user_input:
        # Call the RAG solution via main.py API (for demo purposes we assume an endpoint)
        # You would need to implement the RAG logic in main.py
        answer = get_response(user_input)
        st.write(answer)

    else:
        st.write("Please enter a question.")
