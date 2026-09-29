"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """
    OpenAI ChatCompletion provider.

    Usage:
        llm = OpenAILLM()                          # uses gpt-4o-mini by default
        llm = OpenAILLM(model="gpt-4o")            # specify model
        llm = OpenAILLM(api_key="sk-...")           # explicit key

        response = llm.chat([{"role": "user", "content": "Hello!"}])
        print(response.content)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Either pass `api_key=` or set "
                "the OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=resolved_key)

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Send messages to OpenAI and get a text response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        choice = response.choices[0].message
        return LLMResponse(
            content=choice.content or "",
            tool_calls=[],
            raw=response,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """
        Send messages + tool definitions to OpenAI.

        If the LLM decides to call a tool, the response will have `tool_calls`
        populated instead of (or in addition to) `content`.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        parsed_tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                parsed_tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )

        return LLMResponse(
            content=choice.content or "",
            tool_calls=parsed_tool_calls,
            raw=response,
        )
