"""
OpenAI LLM provider implementation.

This is the concrete implementation that talks to the OpenAI ChatCompletion API.

HOW THIS MAPS TO LANGCHAIN:
  - This is equivalent to LangChain's `ChatOpenAI` class
  - `chat()` calls the API without tools → like `ChatOpenAI.invoke()`
  - `chat_with_tools()` passes function schemas → like `ChatOpenAI.bind_tools().invoke()`
  - We manually parse `response.choices[0].message.tool_calls` into our clean format

WHAT YOU LEARN HERE:
  1. How to construct OpenAI API requests with the `openai` SDK
  2. How tool_calls come back in the API response
  3. How frameworks parse tool_calls into an internal representation
"""

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

        # Resolve API key: explicit arg > env var
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Either pass `api_key=` or set "
                "the OPENAI_API_KEY environment variable."
            )

        # Create the OpenAI client (openai SDK v1+)
        self._client = OpenAI(api_key=resolved_key)

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Send messages to OpenAI and get a text response.

        Under the hood this calls:
            client.chat.completions.create(model=..., messages=..., ...)
        """
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

        HOW TOOL CALLS WORK IN THE API:
        1. We send tool schemas in OpenAI's format:
           [{"type": "function", "function": {"name": ..., "parameters": ...}}]
        2. The API response's `message.tool_calls` contains:
           [{"id": "call_xxx", "function": {"name": "...", "arguments": "{...}"}}]
        3. We parse the JSON `arguments` string into a Python dict
        4. We return a clean list: [{"id": ..., "name": ..., "arguments": {...}}]
        """
        # Only pass tools if we actually have some
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # let the LLM decide

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        # Parse tool calls from OpenAI's format into our clean format
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
