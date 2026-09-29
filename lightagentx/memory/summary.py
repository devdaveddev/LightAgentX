"""
Summary Memory — compresses older messages into a running summary using the LLM.

HOW THIS MAPS TO LANGCHAIN:
  - This is LangChain's `ConversationSummaryMemory`
  - When history exceeds max_messages, older messages are compressed
  - The summary is prepended to the system prompt

WHAT YOU LEARN HERE:
  This is a smarter memory strategy: instead of losing old context, we ask the
  LLM to summarize it. This preserves important information while keeping token
  counts manageable.

  Trade-off: uses extra LLM calls for summarization.
"""

from __future__ import annotations

from typing import Any

from .base import BaseMemory

# We import BaseLLM by string to avoid circular imports at module level
# The actual LLM instance is passed in at __init__ time


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

        # Check if we need to compress
        if len(self._messages) > self.max_messages:
            self._compress()

    def _compress(self) -> None:
        """
        Summarize the older half of messages and keep the recent half.

        This is the key mechanism:
        1. Split messages into old (to summarize) and recent (to keep)
        2. Ask the LLM to produce a summary of old messages + existing summary
        3. Replace old messages with the summary
        """
        split_point = len(self._messages) // 2
        old_messages = self._messages[:split_point]
        recent_messages = self._messages[split_point:]

        # Format old messages for the summarizer
        formatted = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '[tool call]')}"
            for m in old_messages
        )

        prompt = _SUMMARIZE_PROMPT.format(
            existing_summary=self._summary or "(no existing summary)",
            new_messages=formatted,
        )

        # Call the LLM to generate a summary
        response = self._llm.chat([{"role": "user", "content": prompt}])
        self._summary = response.content.strip()

        # Keep only recent messages
        self._messages = recent_messages

    def get_messages(self) -> list[dict[str, Any]]:
        """
        Return messages with the summary injected into the system prompt.

        If there's an existing summary, it gets appended to the system message
        so the LLM has context about earlier conversation.
        """
        result = []

        if self._system_message:
            sys_msg = dict(self._system_message)  # copy
            if self._summary:
                sys_msg["content"] = (
                    f"{sys_msg['content']}\n\n"
                    f"CONVERSATION SUMMARY SO FAR:\n{self._summary}"
                )
            result.append(sys_msg)
        elif self._summary:
            # No system message, but we have a summary — add it as system
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
