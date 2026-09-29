"""
Base tool abstraction and @tool decorator.

HOW THIS MAPS TO LANGCHAIN:
  - `BaseTool` ≈ LangChain's `BaseTool` / `StructuredTool`
  - `@tool` decorator ≈ LangChain's `@tool` decorator
  - JSON Schema generation ≈ what LangChain does with Pydantic models

WHAT YOU LEARN HERE:
  1. How Python type hints get converted into JSON Schema for the OpenAI API
  2. How a decorator can auto-extract function metadata (name, docstring, params)
  3. How frameworks make it dead simple to turn any function into an API-callable tool
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> str:
    """Convert a Python type to its JSON Schema equivalent."""
    return _TYPE_MAP.get(py_type, "string")


def _extract_parameters_schema(func: Callable) -> dict[str, Any]:
    """
    Extract a JSON Schema `parameters` object from a function's type hints.

    This is the magic that makes `@tool` work:
    1. Inspect the function's type hints
    2. Skip `return` annotation
    3. Build a JSON Schema with property names, types, and required list
    4. Parse Google-style docstring for parameter descriptions

    Example:
        def add(a: int, b: int) -> int:
            '''Add two numbers.

            Args:
                a: First number.
                b: Second number.
            '''

        → {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number."},
                "b": {"type": "integer", "description": "Second number."}
            },
            "required": ["a", "b"]
        }
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    # Parse param descriptions from docstring (supports Google-style)
    param_descriptions = _parse_docstring_params(func.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(py_type)

        prop: dict[str, Any] = {"type": json_type}

        # Add description from docstring if available
        if param_name in param_descriptions:
            prop["description"] = param_descriptions[param_name]

        properties[param_name] = prop

        # If no default value, it's required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """
    Parse Google-style docstring to extract parameter descriptions.

    Handles format like:
        Args:
            param_name: Description of the parameter.
            other_param: Another description.
    """
    params: dict[str, str] = {}
    in_args = False

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        elif stripped.lower().startswith(("returns:", "raises:", "example")):
            in_args = False
            continue

        if in_args and ":" in stripped:
            # Parse "param_name: description" or "param_name (type): description"
            name_part, _, desc = stripped.partition(":")
            name = name_part.strip().split("(")[0].strip()
            if name:
                params[name] = desc.strip()

    return params


@dataclass
class BaseTool:
    """
    A tool that an agent can use.

    Attributes:
        name: Unique identifier for the tool.
        description: What the tool does (shown to the LLM).
        parameters: JSON Schema describing the tool's input parameters.
        func: The actual callable that performs the tool's work.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    func: Callable = field(default=lambda **kwargs: None)

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        return self.func(**kwargs)

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert this tool to OpenAI's function-calling schema format.

        Returns:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { JSON Schema }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(func: Callable) -> BaseTool:
    """
    Decorator that converts a regular Python function into a BaseTool.

    It auto-extracts:
      - name → from function name
      - description → from docstring's first line
      - parameters → from type hints + docstring

    Usage:
        @tool
        def calculate(expression: str) -> str:
            '''Evaluate a mathematical expression.

            Args:
                expression: The math expression to evaluate.
            '''
            return str(eval(expression))

        # `calculate` is now a BaseTool instance
        print(calculate.name)           # "calculate"
        print(calculate.description)    # "Evaluate a mathematical expression."
        print(calculate("2 + 2"))       # "4"
    """
    # Extract metadata
    name = func.__name__
    docstring = func.__doc__ or ""
    description = docstring.strip().split("\n")[0] if docstring else name
    parameters = _extract_parameters_schema(func)

    return BaseTool(
        name=name,
        description=description,
        parameters=parameters,
        func=func,
    )
