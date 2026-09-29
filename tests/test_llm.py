"""Tests for the LLM abstraction layer."""

from lightagentx.llm.base import BaseLLM, LLMResponse


class TestLLMResponse:
    def test_empty_response(self):
        r = LLMResponse(content="Hello")
        assert r.content == "Hello"
        assert r.tool_calls == []
        assert r.has_tool_calls is False

    def test_response_with_tool_calls(self):
        r = LLMResponse(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "calc", "arguments": {"x": 1}}
            ],
        )
        assert r.has_tool_calls is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0]["name"] == "calc"


class TestBaseLLMInterface:
    def test_cannot_instantiate_abstract(self):
        """BaseLLM is abstract — can't instantiate directly."""
        try:
            BaseLLM(model="test")
            assert False, "Should have raised TypeError"
        except TypeError:
            pass
