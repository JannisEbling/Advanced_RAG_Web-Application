# app/app.py
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.pipelines.get_answer import invoke_single


def main():
    st.title("RAG Machine Learning")

    # Get user input (query)
    user_input = st.text_input("Enter your question:")

    if user_input:
        # Display a loading spinner while invoking the pipeline
        with st.spinner("Fetching answer..."):
            result = invoke_single(user_input)

        # Check if result is returned
        if result:
            # Assuming the result contains response, reranked_documents, and hallucination_state
            response = result.get("response", "No response generated.")
            reranked_documents = result.get("reranked_documents", [])
            hallucination_state = result.get("hallucination_state", False)

            # Display the results
            st.subheader("Response")
            st.write(response)

            st.subheader("Documents")
            if reranked_documents:
                for doc in reranked_documents:
                    st.write(doc)
            else:
                st.write("No documents found.")

            st.subheader("Hallucination State")
            st.write(
                "Hallucination detected"
                if hallucination_state
                else "No hallucination detected"
            )


if __name__ == "__main__":
    main()
