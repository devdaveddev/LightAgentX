"""Buffer Memory — stores the last N messages in a simple FIFO queue."""

from __future__ import annotations

from typing import Any

from .base import BaseMemory


class BufferMemory(BaseMemory):
    """
    Fixed-size sliding window memory.

    Keeps a system prompt (if set) + the last `max_messages` conversation
    messages. When the buffer is full, the oldest non-system message is dropped.

    Usage:
        memory = BufferMemory(max_messages=20)
        memory.add_message("system", "You are a helpful assistant.")
        memory.add_message("user", "Hello!")
        memory.add_message("assistant", "Hi there!")
        messages = memory.get_messages()
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._system_message: dict[str, Any] | None = None
        self._messages: list[dict[str, Any]] = []

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """
        Add a message. System messages are stored separately so they're
        never evicted by the sliding window.
        """
        message: dict[str, Any] = {"role": role, "content": content}

        if "tool_call_id" in kwargs:
            message["tool_call_id"] = kwargs["tool_call_id"]
        if "tool_calls" in kwargs:
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            tc["arguments"]
                            if isinstance(tc["arguments"], str)
                            else __import__("json").dumps(tc["arguments"])
                        ),
                    },
                }
                for tc in kwargs["tool_calls"]
            ]

        if role == "system":
            self._system_message = message
            return

        self._messages.append(message)

        while len(self._messages) > self.max_messages:
            self._messages.pop(0)

    def get_messages(self) -> list[dict[str, Any]]:
        """Return system prompt + conversation messages."""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        """Clear all messages including system prompt."""
        self._system_message = None
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages) + (1 if self._system_message else 0)

    def __repr__(self) -> str:
        return (
            f"BufferMemory(max_messages={self.max_messages}, "
            f"current={len(self._messages)})"
        )
