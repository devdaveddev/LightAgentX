"""Tool Executor — dispatches tool calls from the LLM to the actual functions."""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry
from ..utils.logger import AgentLogger


class ToolExecutor:
    """
    Executes tool calls by looking them up in a ToolRegistry and invoking them.

    Usage:
        executor = ToolExecutor(registry)
        result = executor.execute({"id": "call_123", "name": "calculate", "arguments": {"expression": "2+2"}})
        print(result)  # "4"
    """

    def __init__(self, registry: ToolRegistry, logger: AgentLogger | None = None):
        self.registry = registry
        self.logger = logger or AgentLogger(verbose=False)

    def execute(self, tool_call: dict[str, Any]) -> str:
        """
        Execute a single tool call.

        Args:
            tool_call: Dict with keys: "id", "name", "arguments"
                       (as returned by LLMResponse.tool_calls)

        Returns:
            String result of the tool execution, or an error message.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        call_id = tool_call.get("id", "unknown")

        self.logger.action(name, arguments)

        tool = self.registry.get(name)
        if tool is None:
            error_msg = (
                f"Tool '{name}' not found. "
                f"Available tools: {self.registry.tool_names}"
            )
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

        try:
            result = tool(**arguments)
            result_str = str(result) if result is not None else "Done (no output)"
            self.logger.observation(result_str)
            return result_str

        except Exception as e:
            error_msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

    def execute_many(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Execute multiple tool calls and return results paired with their IDs.

        Returns:
            List of {"tool_call_id": "...", "content": "..."} dicts,
            ready to be added as tool messages to the conversation.
        """
        results = []
        for tc in tool_calls:
            result = self.execute(tc)
            results.append(
                {
                    "tool_call_id": tc.get("id", "unknown"),
                    "content": result,
                }
            )
        return results
