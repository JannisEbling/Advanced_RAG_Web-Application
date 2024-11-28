# app/app.py
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.pipelines.get_answer import invoke


def main():
    st.title("RAG Machine Learning")

    # Introduction
    st.markdown("""
    Welcome to this advanced Machine Learning Query System! This application uses Retrieval-Augmented Generation (RAG) 
    to provide accurate answers to your machine learning questions. It searches through comprehensive machine learning textbooks 
    and research papers, utilizing intelligent document retrieval and multi-agent processing to deliver precise, 
    well-supported responses.
    """)

    # Get user input (query)
    user_input = st.text_input("Enter your question:")

    if user_input:
        # Display a loading spinner while invoking the pipeline
        with st.spinner("Fetching answer..."):
            result = invoke(user_input)

        # Check if result is returned
        if result:
            # Assuming the result contains response, reranked_documents, and hallucination_state
            response = result.get("response", "No response generated.")
            reranked_documents = result.get("reranked_documents", [])
            figure_paths = result.get("figure_paths", [])
            confidence_score = result.get("response_confidence", 0.0)
            print(figure_paths)

            # Display the results
            st.subheader("Response")

            # Display confidence score with color coding
            confidence_color = (
                "green"
                if confidence_score >= 0.7
                else "orange"
                if confidence_score >= 0.4
                else "red"
            )
            st.markdown(
                f"<span style='color: {confidence_color}'>Confidence Score: {confidence_score:.2f}</span>",
                unsafe_allow_html=True,
            )

            st.write(response)

            # Display referenced figures if any
            if figure_paths:
                st.subheader("Referenced Figures")
                # Create columns for figures
                cols = st.columns(min(len(figure_paths), 3))  # Max 3 columns

                for idx, fig_path in enumerate(figure_paths):
                    col = cols[idx % 3]  # Cycle through columns
                    try:
                        # Extract filename from path
                        filename = Path(fig_path).name
                        # Display figure in column with filename as caption
                        with col:
                            st.image(fig_path, caption=filename, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading figure {filename}: {str(e)}")

            # Display reranked documents if available
            if reranked_documents:
                st.subheader("Related Documents")
                for doc in reranked_documents:
                    # Display metadata (relevance score and page number)
                    metadata_text = [
                        f"**Relevance Score:** {doc.metadata.get('relevance_score', 'N/A')}"
                    ]
                    if "page_number" in doc.metadata:
                        metadata_text.append(f"**Page:** {doc.metadata['page_number']}")
                    st.markdown(" | ".join(metadata_text))

                    # Display document content
                    st.markdown(doc.page_content)
                    st.divider()


if __name__ == "__main__":
    main()
