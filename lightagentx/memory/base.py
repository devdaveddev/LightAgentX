"""Abstract base class for memory modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """
    Abstract base for all memory implementations.

    Every memory module must store messages in the OpenAI format::

        {"role": "system"|"user"|"assistant"|"tool", "content": "..."}
    """

    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """
        Add a message to memory.

        Args:
            role: One of "system", "user", "assistant", "tool".
            content: The message content.
            **kwargs: Extra metadata (e.g., tool_call_id for tool messages).
        """
        ...

    @abstractmethod
    def get_messages(self) -> list[dict[str, Any]]:
        """
        Return the current conversation history as a list of message dicts.

        Returns:
            List of message dicts in OpenAI format.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored messages."""
        ...

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """Convenience method to add a tool response message."""
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def add_assistant_tool_calls(self, content: str, tool_calls: list[dict]) -> None:
        """
        Add an assistant message that contains tool calls.

        This is needed because when the LLM requests tool calls, the assistant
        message must include the tool_calls field for the API conversation flow.
        """
        self.add_message("assistant", content, tool_calls=tool_calls)
