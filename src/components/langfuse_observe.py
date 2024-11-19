from langfuse.decorators import observe
from langfuse.openai import openai

@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful customer care assistant that can classify incoming messages and create a response.",
            },
            {"role": "user", "content": "Hi there, I have a question about my bill. Can you help me?"},
        ],
    )