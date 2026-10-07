"""Base tool abstraction and @tool decorator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, get_type_hints

from ..sandbox.policy import Risk


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

    Inspects the function's type hints and Google-style docstring to build
    a JSON Schema with property names, types, descriptions, and required list.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    param_descriptions = _parse_docstring_params(func.__doc__ or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(py_type)

        prop: dict[str, Any] = {"type": json_type}

        if param_name in param_descriptions:
            prop["description"] = param_descriptions[param_name]

        properties[param_name] = prop

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
        risk: How dangerous the tool is (LOW/MEDIUM/HIGH). None = undeclared;
            a sandboxed agent then treats it as the policy's
            `undeclared_tool_risk` (HIGH by default).
        reads: Names of arguments that are file paths the tool reads.
        writes: Names of arguments that are file paths the tool writes.
        guarded: True if the tool already checks the sandbox itself
            (the built-in SmartOS tools), so the agent must not check twice.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    func: Callable = field(default=lambda **kwargs: None)
    risk: Risk | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    guarded: bool = False

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        return self.func(**kwargs)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert this tool to OpenAI's function-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _as_risk(risk: Risk | str | None) -> Risk | None:
    if risk is None or isinstance(risk, Risk):
        return risk
    try:
        return Risk[str(risk).upper()]
    except KeyError:
        raise ValueError(f"risk must be 'low', 'medium' or 'high', got {risk!r}") from None


def tool(
    func: Callable | None = None,
    *,
    risk: Risk | str | None = None,
    reads: Iterable[str] = (),
    writes: Iterable[str] = (),
) -> Any:
    """
    Decorator that converts a regular Python function into a BaseTool.

    Auto-extracts name, description, and parameters from the function's
    name, docstring, and type hints.

    Usage:
        @tool
        def calculate(expression: str) -> str:
            '''Evaluate a mathematical expression.

            Args:
                expression: The math expression to evaluate.
            '''
            return str(eval(expression))

        # With a safety declaration (used when the agent has a sandbox):
        @tool(risk="high", writes=["path"])
        def delete_file(path: str) -> str:
            ...
    """

    def build(f: Callable) -> BaseTool:
        docstring = f.__doc__ or ""
        parameters = _extract_parameters_schema(f)
        declared = set(parameters.get("properties", {}))
        for arg in (*reads, *writes):
            if arg not in declared:
                raise ValueError(f"Tool '{f.__name__}' declares path argument '{arg}' "
                                 f"but has no such parameter.")
        return BaseTool(
            name=f.__name__,
            description=docstring.strip().split("\n")[0] if docstring else f.__name__,
            parameters=parameters,
            func=f,
            risk=_as_risk(risk),
            reads=tuple(reads),
            writes=tuple(writes),
        )

    return build(func) if func is not None else build


def mark_guarded(tools: list[BaseTool]) -> list[BaseTool]:
    """Flag tools that perform their own sandbox checks (avoids double prompts)."""
    for t in tools:
        t.guarded = True
    return tools
