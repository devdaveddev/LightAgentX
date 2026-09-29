"""
Tool Registry — a name-based lookup table for tools.

HOW THIS MAPS TO LANGCHAIN:
  - LangChain stores tools in a list and does name-based lookup internally
  - The `to_openai_schema()` method is what LangChain calls when it does
    `format_tool_to_openai_function()` / `convert_to_openai_tool()`

WHAT YOU LEARN HERE:
  The registry is just a dict mapping tool names → tool objects.
  Its main job is to bulk-convert all tools into the JSON Schema format
  that the OpenAI API expects.
"""

from __future__ import annotations

from typing import Any

from .base import BaseTool


class ToolRegistry:
    """
    A registry that maps tool names to BaseTool instances.

    Usage:
        registry = ToolRegistry()
        registry.register(my_tool)
        registry.register(another_tool)

        # Get OpenAI-compatible schemas for all tools
        schemas = registry.to_openai_schema()

        # Look up a tool by name
        tool = registry.get("my_tool")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises ValueError if name already taken."""
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Each tool must have a unique name."
            )
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """
        Convert ALL registered tools to OpenAI function-calling format.

        Returns a list like:
        [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a math expression.",
                    "parameters": { ... }
                }
            },
            ...
        ]
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"
