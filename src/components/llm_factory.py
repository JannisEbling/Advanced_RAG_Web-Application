from typing import Any, Dict, List, Type

import instructor
from anthropic import Anthropic
from config.settings import get_settings
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field
from src.secure.secrets import secrets


class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
            "azure": lambda s: instructor.from_openai(
                AzureOpenAI(
                    api_key=secrets.get_secret("AZURE_OPENAI_API_KEY"),
                    api_version=s.api_version,
                    azure_endpoint=secrets.get_secret("AZURE_OPENAI_ENDPOINT"),
                )
            ),
            "anthropic": lambda s: instructor.from_anthropic(
                Anthropic(api_key=s.api_key)
            ),
            "llama": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }

        # For Azure OpenAI, we need to use deployment_id instead of model
        if self.provider == "azure":
            completion_params["deployment_id"] = completion_params.pop("model")

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
