"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """
    Unified response from any LLM provider.

    Attributes:
        content: The text response (may be empty if tool_calls are present).
        tool_calls: List of tool call requests from the LLM.
                    Each dict has: {"id": str, "name": str, "arguments": dict}
        raw: The raw response object from the provider.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement:
      - chat(messages) → LLMResponse
      - chat_with_tools(messages, tools) → LLMResponse

    The `messages` format follows OpenAI's convention:
      [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]

    The `tools` format follows OpenAI's function-calling schema:
      [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Send a list of messages and get a text completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            LLMResponse with the text in `.content`.
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """
        Send messages along with tool definitions. The LLM may choose to
        call a tool instead of (or in addition to) responding with text.

        Args:
            messages: Conversation history.
            tools: Tool definitions in OpenAI function-calling format.

        Returns:
            LLMResponse — check `.has_tool_calls` to see if tools were invoked.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
