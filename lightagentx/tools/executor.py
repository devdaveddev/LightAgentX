"""Tool Executor — dispatches tool calls from the LLM to the actual functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .registry import ToolRegistry
from ..utils.logger import AgentLogger

if TYPE_CHECKING:
    from ..sandbox import Sandbox


class ToolExecutor:
    """
    Executes tool calls by looking them up in a ToolRegistry and invoking them.

    Usage:
        executor = ToolExecutor(registry)
        result = executor.execute({"id": "call_123", "name": "calculate", "arguments": {"expression": "2+2"}})
        print(result)  # "4"
    """

    def __init__(self, registry: ToolRegistry, logger: AgentLogger | None = None,
                 sandbox: "Sandbox | None" = None):
        self.registry = registry
        self.logger = logger or AgentLogger(verbose=False)
        self.sandbox = sandbox

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
            if self.sandbox is not None and not tool.guarded:
                arguments = self._gate(tool, dict(arguments))
            result = tool(**arguments)
            result_str = str(result) if result is not None else "Done (no output)"
            self.logger.observation(result_str)
            return result_str

        except Exception as e:
            error_msg = f"Tool '{name}' failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"

    def _gate(self, tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        The sandbox checkpoint every unguarded tool call passes through.

        1. Declared path arguments are checked against the policy and replaced
           by the resolved absolute path that was approved, so the tool opens
           exactly what was checked (a relative path can't point elsewhere).
        2. The call is authorized at the tool's risk level: low risk runs,
           risky calls need a human "yes". Every decision is audit-logged.

        Raises SandboxViolation, which `execute` turns into an error the LLM reads.
        """
        sandbox = self.sandbox
        for arg in tool.reads:
            if arguments.get(arg) is not None:
                arguments[arg] = str(sandbox.check_read(arguments[arg]))
        for arg in tool.writes:
            if arguments.get(arg) is not None:
                arguments[arg] = str(sandbox.check_write(arguments[arg]))
        risk = tool.risk if tool.risk is not None else sandbox.policy.undeclared_tool_risk
        detail = json.dumps(arguments, default=str)[:300]
        if tool.risk is None:
            detail += "  (risk not declared)"
        sandbox.authorize(f"tool:{tool.name}", detail, risk)
        return arguments

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
