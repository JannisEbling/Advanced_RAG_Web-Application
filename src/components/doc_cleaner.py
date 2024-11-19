import re

import fitz
from langchain.schema import Document

from src import logger, DocumentProcessingError


def replace_t_with_space(text):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        text (str):  The input text string.

    Returns:
        cleaned_text (str): The modified text with tab characters replaced by spaces.
    """

    cleaned_text = text.replace("\t", " ")
    return cleaned_text


def split_into_chapters(pdf_path):
    """
    Splits a PDF into chapters based on chapter title patterns and formatting using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter name as metadata.
    """

    pdf_document = fitz.open(pdf_path)
    chapter_docs = []
    chapter_text = ""
    chapter_name = "None"

    # Regex pattern: Ask Chat GPT how it works :P
    chapter_pattern = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][\w\s\-:]*[A-Za-z]+$")

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:

            text = block[4].strip()

            if chapter_pattern.match(text):
                if chapter_text:
                    doc = Document(
                        page_content=chapter_text, metadata={"chapter": chapter_name}
                    )
                    chapter_docs.append(doc)
                chapter_name = text
                chapter_text = ""
            else:
                chapter_text += text + "\n"

    if chapter_text:
        doc = Document(page_content=chapter_text, metadata={"chapter": chapter_name})
        chapter_docs.append(doc)

    pdf_document.close()
    return chapter_docs


def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text (str): The input text string.

    Returns:
        cleaned_text (str): The modified text string with double newlines replaced by single newlines.
    """

    cleaned_text = re.sub(r"\n\n", "\n", text)
    return cleaned_text


def replace_new_line_with_space(text):
    """
    Replaces newline characters ('\n\n') with a space (' ').

    Args:
        text (str): The input text string.

    Returns:
        cleaned_text (str): The text string with newlines replaced by spaces.
    """

    cleaned_text = re.sub(r"\n", " ", text)
    return cleaned_text


def remove_substrings(text, substrings):
    """
    Removes all occurrences of each substring in the substrings list from the text.

    Args:
        text (str): The input text from which substrings will be removed.
        substrings (list): A list of strings to remove from the text.

    Returns:
        str: The cleaned text with specified substrings removed.
    """
    for substring in substrings:
        text = text.replace(substring, "")
    return text


def filter_recurrent_obsolescences_with_remove(text):
    """
    Replaces recurrent and non-informative strings from a text

    Args:
        text (str): The input text string.

    Returns:
        cleaned_text (str): The text string with removed non-informative parts.
    """
    substrings = [
        " The Author(s), under exclusive license to Springer Nature Switzerland AG 2024 \nC. M. Bishop, H. Bishop, Deep Learning, https://doi.org/10.1007/978-3-031-45468-4_"
    ]
    cleaned_text = remove_substrings(text, substrings)
    return cleaned_text


class DocumentCleaner:
    """Cleans and preprocesses documents for the RAG pipeline."""

    def clean_documents(self, documents):
        """
        Clean and preprocess a list of documents.

        Args:
            documents: List of documents to clean

        Returns:
            List of cleaned documents

        Raises:
            DocumentProcessingError: If document cleaning fails
        """
        if not documents:
            raise DocumentProcessingError(
                "Empty document list provided for cleaning",
                details={"num_docs": 0}
            )

        try:
            logger.info("Cleaning %d documents", len(documents))
            cleaned_docs = []
            
            for doc in documents:
                cleaned_content = self._clean_text(doc.page_content)
                if cleaned_content.strip():  # Only keep non-empty documents
                    cleaned_doc = Document(
                        page_content=cleaned_content,
                        metadata=doc.metadata
                    )
                    cleaned_docs.append(cleaned_doc)
            
            logger.info("Successfully cleaned %d documents", len(cleaned_docs))
            return cleaned_docs

        except Exception as e:
            raise DocumentProcessingError(
                "Failed to clean documents",
                details={
                    "num_docs": len(documents),
                    "error": str(e)
                }
            )

    def _clean_text(self, text):
        """
        Clean a single text string.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        try:
            # Remove excessive whitespace
            cleaned = " ".join(text.split())
            
            # Remove very short lines (likely noise)
            lines = [line for line in cleaned.splitlines() if len(line.strip()) > 10]
            cleaned = "\n".join(lines)
            
            return cleaned

        except Exception as e:
            raise DocumentProcessingError(
                "Failed to clean text",
                details={
                    "text_length": len(text),
                    "error": str(e)
                }
            )
