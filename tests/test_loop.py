"""Tests for the agent loop engine using a mock LLM."""

from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.tools.base import tool
from lightagentx.loop.engine import AgentLoop


class MockLLM(BaseLLM):
    """
    A mock LLM that returns pre-configured responses.

    This is how you test agent systems without making API calls.
    """

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(model="mock")
        self._responses = list(responses)
        self._call_count = 0

    def chat(self, messages):
        return self._next_response()

    def chat_with_tools(self, messages, tools):
        return self._next_response()

    def _next_response(self):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return LLMResponse(content="No more mock responses")


class TestAgentLoop:
    def test_simple_response_no_tools(self):
        """LLM answers directly without calling any tools."""
        mock = MockLLM([
            LLMResponse(content="The answer is 42."),
        ])

        loop = AgentLoop(llm=mock, verbose=False)
        result = loop.run("What is the meaning of life?")
        assert result == "The answer is 42."

    def test_tool_call_then_response(self):
        """LLM calls a tool, then uses the result to answer."""
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        mock = MockLLM([
            # First call: LLM wants to use the multiply tool
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "multiply",
                    "arguments": {"a": 25, "b": 37},
                }],
            ),
            # Second call: LLM uses the result to form an answer
            LLMResponse(content="25 * 37 = 925"),
        ])

        loop = AgentLoop(llm=mock, tools=[multiply], verbose=False)
        result = loop.run("What is 25 * 37?")
        assert "925" in result

    def test_multiple_tool_calls(self):
        """LLM makes multiple tool calls in one round."""
        @tool
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        mock = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "double", "arguments": {"x": 5}},
                    {"id": "c2", "name": "double", "arguments": {"x": 10}},
                ],
            ),
            LLMResponse(content="5 doubled is 10, and 10 doubled is 20."),
        ])

        loop = AgentLoop(llm=mock, tools=[double], verbose=False)
        result = loop.run("Double 5 and 10")
        assert "10" in result
        assert "20" in result

    def test_max_iterations_guard(self):
        """Loop stops after max_iterations even if LLM keeps calling tools."""
        @tool
        def noop(x: str) -> str:
            """Do nothing."""
            return x

        # LLM always wants to call a tool — never gives a final answer
        infinite_tool_calls = [
            LLMResponse(
                content="",
                tool_calls=[{"id": f"c{i}", "name": "noop", "arguments": {"x": "hi"}}],
            )
            for i in range(20)
        ]

        mock = MockLLM(infinite_tool_calls)
        loop = AgentLoop(llm=mock, tools=[noop], max_iterations=3, verbose=False)
        result = loop.run("Loop forever")
        assert "unable to complete" in result.lower() or "3" in result
