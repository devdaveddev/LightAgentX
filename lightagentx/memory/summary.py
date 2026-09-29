"""Summary Memory — compresses older messages into a running summary using the LLM."""

from __future__ import annotations

from typing import Any

from .base import BaseMemory


_SUMMARIZE_PROMPT = """You are a conversation summarizer. Given the existing summary and new messages, produce an updated concise summary that captures all important information, decisions, and context.

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES:
{new_messages}

UPDATED SUMMARY:"""


class SummaryMemory(BaseMemory):
    """
    Memory that compresses old messages into a running LLM-generated summary.

    When the message buffer exceeds `max_messages`, the oldest messages are
    summarized and the summary is stored. New messages continue to accumulate
    until the next compression cycle.

    Usage:
        from lightagentx.llm import OpenAILLM

        llm = OpenAILLM()
        memory = SummaryMemory(llm=llm, max_messages=10)
        memory.add_message("system", "You are helpful.")
        # ... add many messages ...
        # When > 10 messages, older ones get auto-summarized
    """

    def __init__(self, llm: Any, max_messages: int = 10):
        """
        Args:
            llm: An LLM instance (BaseLLM subclass) used for summarization.
            max_messages: Max messages before triggering compression.
        """
        self._llm = llm
        self.max_messages = max_messages
        self._system_message: dict[str, Any] | None = None
        self._messages: list[dict[str, Any]] = []
        self._summary: str = ""

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
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

        if len(self._messages) > self.max_messages:
            self._compress()

    def _compress(self) -> None:
        """Summarize the older half of messages and keep the recent half."""
        split_point = len(self._messages) // 2
        old_messages = self._messages[:split_point]
        recent_messages = self._messages[split_point:]

        formatted = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '[tool call]')}"
            for m in old_messages
        )

        prompt = _SUMMARIZE_PROMPT.format(
            existing_summary=self._summary or "(no existing summary)",
            new_messages=formatted,
        )

        response = self._llm.chat([{"role": "user", "content": prompt}])
        self._summary = response.content.strip()

        self._messages = recent_messages

    def get_messages(self) -> list[dict[str, Any]]:
        """Return messages with the summary injected into the system prompt."""
        result = []

        if self._system_message:
            sys_msg = dict(self._system_message)
            if self._summary:
                sys_msg["content"] = (
                    f"{sys_msg['content']}\n\n"
                    f"CONVERSATION SUMMARY SO FAR:\n{self._summary}"
                )
            result.append(sys_msg)
        elif self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"CONVERSATION SUMMARY SO FAR:\n{self._summary}",
                }
            )

        result.extend(self._messages)
        return result

    def clear(self) -> None:
        self._system_message = None
        self._messages.clear()
        self._summary = ""

    @property
    def summary(self) -> str:
        """Access the current running summary."""
        return self._summary

    def __repr__(self) -> str:
        return (
            f"SummaryMemory(max_messages={self.max_messages}, "
            f"current={len(self._messages)}, "
            f"has_summary={bool(self._summary)})"
        )
