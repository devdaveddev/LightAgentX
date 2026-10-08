"""Pick an LLM from the environment — shared by the `lightx` command-line tools."""

from __future__ import annotations

import os

from .base import BaseLLM


class NoLLMConfigured(RuntimeError):
    pass


def llm_from_env(provider: str = "auto", model: str | None = None, base_url: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> BaseLLM:
    """
    Build an LLM client.

    provider "auto" picks, in order: an OpenAI-compatible `base_url` (e.g. a local
    Ollama server), then whichever of ANTHROPIC_API_KEY, OPENAI_API_KEY or
    GOOGLE_API_KEY is set.
    """
    if provider == "auto":
        if base_url:
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            raise NoLLMConfigured(
                "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY, "
                "or point --base-url at an OpenAI-compatible local server (e.g. Ollama)."
            )
    kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    if model:
        kwargs["model"] = model
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        return AnthropicLLM(**kwargs)
    if provider == "gemini":
        from .gemini_llm import GeminiLLM
        return GeminiLLM(**kwargs)
    from .openai_llm import OpenAILLM
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "local")
    return OpenAILLM(**kwargs)
