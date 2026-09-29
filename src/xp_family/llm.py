"""LLM clients. One shared Gemini client replaces the three copies in the original views."""

from __future__ import annotations

from typing import Protocol

from xp_family.config import Settings
from xp_family.observability import trace


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted without an API key."""


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, *, json_mode: bool = False) -> str: ...


class GeminiLLM:
    """Gemini via the current `google-genai` SDK, with every call traced."""

    def __init__(self, settings: Settings) -> None:
        if settings.gemini_api_key is None:
            raise LLMNotConfiguredError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.name = settings.gemini_model
        self._temperature = settings.llm_temperature

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        with trace(
            "llm_call", provider="gemini", model=self.name, prompt_chars=len(prompt)
        ) as span:
            response = self._client.models.generate_content(
                model=self.name, contents=prompt, config=config
            )
            usage = response.usage_metadata
            span.set(
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )
        return (response.text or "").strip()


class EchoLLM:
    """Deterministic offline stand-in used by tests and keyless demos.

    It answers with the first context block it was given, so the pipeline stays
    grounded and testable without network access.
    """

    name = "echo"

    def __init__(self, reply: str | None = None) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        if self._reply is not None:
            return self._reply
        marker = "[1]"
        start = prompt.find(marker)
        if start == -1:
            return "Je ne dispose pas d'informations suffisantes."
        return prompt[start : start + 400].split("\n\n")[0]
