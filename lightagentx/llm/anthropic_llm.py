"""Anthropic (Claude) LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from .base import BaseLLM, LLMResponse
from .key_guard import SecureKey


class AnthropicLLM(BaseLLM):
    """
    Anthropic Claude provider via the Messages API.

    Requires: ``pip install anthropic``

    Usage:
        llm = AnthropicLLM()                                # claude-3-5-sonnet
        llm = AnthropicLLM(model="claude-3-5-haiku-latest")  # specify model
        llm = AnthropicLLM(api_key="sk-ant-...")             # explicit key

        response = llm.chat([{"role": "user", "content": "Hello!"}])
        print(response.content)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLM. "
                "Install it with: pip install lightagentx[anthropic]"
            )

        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Either pass `api_key=` or set "
                "the ANTHROPIC_API_KEY environment variable."
            )

        self._api_key = SecureKey(resolved_key, provider="anthropic")
        self._client = anthropic.Anthropic(api_key=self._api_key.unwrap())

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Send messages to Anthropic and get a text response."""
        system_prompt, user_messages = self._split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": user_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(content=content, tool_calls=[], raw=response)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """
        Send messages + tool definitions to Anthropic.

        Converts OpenAI-format tool schemas to Anthropic's format, and converts
        Anthropic's tool_use response blocks back to the framework's format.
        """
        system_prompt, user_messages = self._split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(user_messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self._client.messages.create(**kwargs)

        content = ""
        parsed_tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                parsed_tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input if isinstance(block.input, dict) else json.loads(block.input),
                })

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            raw=response,
        )

    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Extract system prompt from messages (Anthropic uses a separate system param)."""
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)
        return system_prompt, user_messages

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI function-calling tool format to Anthropic's format."""
        anthropic_tools = []
        for t in tools:
            func = t.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert OpenAI-style tool result messages to Anthropic's format.

        OpenAI uses role='tool' with tool_call_id; Anthropic uses role='user'
        with a tool_result content block.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", "unknown"),
                        "content": msg.get("content", ""),
                    }],
                })
            elif role == "assistant" and "tool_calls" in msg:
                content_blocks: list[dict[str, Any]] = []
                text = msg.get("content", "")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", "unknown"),
                        "name": func.get("name", ""),
                        "input": json.loads(args) if isinstance(args, str) else args,
                    })
                converted.append({"role": "assistant", "content": content_blocks})
            else:
                converted.append(msg)

        return converted
