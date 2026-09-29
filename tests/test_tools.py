"""Tests for the tools module — @tool decorator, registry, and executor."""

from lightagentx.tools.base import BaseTool, tool
from lightagentx.tools.registry import ToolRegistry
from lightagentx.tools.executor import ToolExecutor


class TestToolDecorator:
    def test_basic_tool(self):
        @tool
        def greet(name: str) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
            """
            return f"Hello, {name}!"

        assert isinstance(greet, BaseTool)
        assert greet.name == "greet"
        assert greet.description == "Say hello to someone."
        assert "name" in greet.parameters["properties"]
        assert greet.parameters["properties"]["name"]["type"] == "string"
        assert "name" in greet.parameters.get("required", [])

    def test_tool_callable(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        result = add(a=3, b=7)
        assert result == 10

    def test_openai_schema(self):
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return query

        schema = search.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web."

    def test_multiple_types(self):
        @tool
        def mixed(name: str, age: int, active: bool) -> str:
            """A tool with mixed types.

            Args:
                name: User name.
                age: User age.
                active: Is active.
            """
            return f"{name}, {age}, {active}"

        props = mixed.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"
        assert props["active"]["type"] == "boolean"


class TestToolRegistry:
    def test_register_and_get(self):
        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        reg = ToolRegistry()
        reg.register(my_tool)
        assert reg.get("my_tool") is my_tool
        assert "my_tool" in reg
        assert len(reg) == 1

    def test_duplicate_registration_fails(self):
        @tool
        def dup_tool(x: str) -> str:
            """Dup."""
            return x

        reg = ToolRegistry()
        reg.register(dup_tool)
        try:
            reg.register(dup_tool)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_openai_schema_bulk(self):
        @tool
        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        @tool
        def tool_b(y: int) -> int:
            """Tool B."""
            return y

        reg = ToolRegistry()
        reg.register_many([tool_a, tool_b])
        schemas = reg.to_openai_schema()
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_b" in names


class TestToolExecutor:
    def test_successful_execution(self):
        @tool
        def adder(a: int, b: int) -> int:
            """Add."""
            return a + b

        reg = ToolRegistry()
        reg.register(adder)
        executor = ToolExecutor(reg)

        result = executor.execute({
            "id": "call_1",
            "name": "adder",
            "arguments": {"a": 3, "b": 4},
        })
        assert result == "7"

    def test_tool_not_found(self):
        reg = ToolRegistry()
        executor = ToolExecutor(reg)

        result = executor.execute({
            "id": "call_1",
            "name": "nonexistent",
            "arguments": {},
        })
        assert "Error" in result
        assert "not found" in result

    def test_tool_execution_error(self):
        @tool
        def bad_tool(x: int) -> int:
            """Fails."""
            raise ValueError("boom!")

        reg = ToolRegistry()
        reg.register(bad_tool)
        executor = ToolExecutor(reg)

        result = executor.execute({
            "id": "call_1",
            "name": "bad_tool",
            "arguments": {"x": 1},
        })
        assert "Error" in result
        assert "boom!" in result

    def test_execute_many(self):
        @tool
        def echo(msg: str) -> str:
            """Echo."""
            return msg

        reg = ToolRegistry()
        reg.register(echo)
        executor = ToolExecutor(reg)

        results = executor.execute_many([
            {"id": "c1", "name": "echo", "arguments": {"msg": "hello"}},
            {"id": "c2", "name": "echo", "arguments": {"msg": "world"}},
        ])
        assert len(results) == 2
        assert results[0]["tool_call_id"] == "c1"
        assert results[0]["content"] == "hello"
        assert results[1]["content"] == "world"
