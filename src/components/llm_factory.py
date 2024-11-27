from typing import Any, Dict, List, Type, Optional

import instructor
from openai import OpenAI, AzureOpenAI
from src.config.settings import get_settings
from pydantic import BaseModel, Field


class LLMFactory:
    def __init__(self, provider: str):
        """Initialize LLM factory with specified provider.

        Args:
            provider: The LLM provider to use ('azure', 'openai', etc.)
        """
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize the appropriate LLM client based on provider."""
        client_initializers = {
            "openai": lambda s: instructor.patch(OpenAI(api_key=s.api_key)),
            "azure": lambda s: instructor.patch(
                AzureOpenAI(
                    api_key=s.api_key,
                    api_version=s.api_version,
                    azure_endpoint=s.api_endpoint,
                    azure_deployment=s.deployment_id,
                )
            ),
            "llama": lambda s: instructor.patch(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if not initializer:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return initializer(self.settings)

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        """Create a completion using the initialized LLM client.

        Args:
            response_model: Pydantic model for response validation
            messages: List of message dictionaries
            **kwargs: Additional parameters to override defaults

        Returns:
            Validated response using the provided response_model
        """
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }

        # Remove None values
        completion_params = {
            k: v for k, v in completion_params.items() if v is not None
        }

        return self.client.chat.completions.create(**completion_params)


if __name__ == "__main__":

    class CompletionModel(BaseModel):
        response: str = Field(description="Your response to the user.")
        reasoning: str = Field(description="Explain your reasoning for the response.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "If it takes 2 hours to dry 1 shirt out in the sun, how long will it take to dry 5 shirts?",
        },
    ]

    llm = LLMFactory("openai")
    completion = llm.create_completion(
        response_model=CompletionModel,
        messages=messages,
    )
    assert isinstance(completion, CompletionModel)

    print(f"Response: {completion.response}\n")
    print(f"Reasoning: {completion.reasoning}")
