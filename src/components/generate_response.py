from typing import List, Dict, Any

from langchain.schema import Document
from langchain_openai import AzureOpenAI

from src import logger, GenerationError
from src.secure.secrets import secrets
from src.prompts.prompt_manager import PromptManager


class ResponseGenerator:
    """Generates responses based on retrieved documents and user queries."""

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the response generator.

        Args:
            model_name: Name of the Azure OpenAI model to use

        Raises:
            GenerationError: If initialization fails
        """
        try:
            self.model_name = model_name
            self.llm = AzureOpenAI(
                azure_deployment=model_name,
                api_version=secrets.get_secret("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=secrets.get_secret("AZURE_OPENAI_ENDPOINT"),
                api_key=secrets.get_secret("AZURE_OPENAI_API_KEY"),
                temperature=0.7,
            )
            self.prompt_manager = PromptManager()
            logger.info("Initialized ResponseGenerator with model %s", model_name)
        except Exception as e:
            raise GenerationError(
                "Failed to initialize ResponseGenerator",
                details={
                    "model": model_name,
                    "error": str(e)
                }
            )

    def generate_response(
        self, 
        query: str, 
        context_docs: List[Document],
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate a response based on the query and context documents.

        Args:
            query: User's query
            context_docs: Retrieved context documents
            chat_history: Optional chat history for context

        Returns:
            Generated response

        Raises:
            GenerationError: If response generation fails
        """
        if not query.strip():
            raise GenerationError(
                "Empty query provided for response generation",
                details={"query_length": 0}
            )

        if not context_docs:
            raise GenerationError(
                "No context documents provided for response generation",
                details={"num_docs": 0}
            )

        try:
            logger.info("Generating response for query")
            context = self._prepare_context(context_docs)
            prompt = self.prompt_manager.get_response_prompt(
                query=query,
                context=context,
                chat_history=chat_history or []
            )
            
            response = self.llm.invoke(prompt)
            logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            raise GenerationError(
                "Failed to generate response",
                details={
                    "query_length": len(query),
                    "num_docs": len(context_docs),
                    "has_history": bool(chat_history),
                    "model": self.model_name,
                    "error": str(e)
                }
            )

    def _prepare_context(self, docs: List[Document]) -> str:
        """
        Prepare context documents for the prompt.

        Args:
            docs: List of context documents

        Returns:
            Formatted context string
        """
        try:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get('source', f'Document {i}')
                context_parts.append(f"[{source}]: {content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            raise GenerationError(
                "Failed to prepare context",
                details={
                    "num_docs": len(docs),
                    "error": str(e)
                }
            )
