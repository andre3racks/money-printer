import os
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate_code(self, prompt: str, system_prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model="gemini-3.1-pro-preview"):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.model = model
        self.types = types

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        return response.text

class AnthropicProvider(LLMProvider):
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_code(self, prompt: str, system_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

def get_llm_provider() -> LLMProvider:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "anthropic":
        return AnthropicProvider()
    return GeminiProvider()
