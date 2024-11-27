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

            # Display the results
            st.subheader("Response")
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
                    st.markdown(
                        f"**Relevance Score:** {doc.metadata.get('relevance_score', 'N/A')}"
                    )
                    st.markdown(doc.page_content)
                    st.divider()


if __name__ == "__main__":
    main()
