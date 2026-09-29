"""Tests for memory modules."""

from lightagentx.memory.buffer import BufferMemory


class TestBufferMemory:
    def test_add_and_get(self):
        mem = BufferMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi!")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_system_message_preserved(self):
        mem = BufferMemory(max_messages=2)
        mem.add_message("system", "You are helpful")
        mem.add_message("user", "msg1")
        mem.add_message("assistant", "resp1")
        mem.add_message("user", "msg2")
        mem.add_message("assistant", "resp2")
        msgs = mem.get_messages()
        # System message + last 2 messages
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 3  # system + 2 recent

    def test_eviction(self):
        mem = BufferMemory(max_messages=3)
        for i in range(5):
            mem.add_message("user", f"message {i}")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "message 2"

    def test_clear(self):
        mem = BufferMemory()
        mem.add_message("system", "sys")
        mem.add_message("user", "hello")
        mem.clear()
        assert mem.get_messages() == []

    def test_tool_messages(self):
        mem = BufferMemory()
        mem.add_tool_message("call_123", "result here")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_123"
        assert msgs[0]["content"] == "result here"

    def test_assistant_tool_calls(self):
        mem = BufferMemory()
        mem.add_assistant_tool_calls(
            content="",
            tool_calls=[{"id": "c1", "name": "calc", "arguments": {"x": 1}}],
        )
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert "tool_calls" in msgs[0]
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "calc"

from lightagentx.memory.summary import SummaryMemory
from lightagentx.llm.base import BaseLLM, LLMResponse

class DummyLLM(BaseLLM):
    def chat(self, messages):
        return LLMResponse(content="Summarized!")
    def chat_with_tools(self, messages, tools):
        return LLMResponse(content="Summarized!")

class TestSummaryMemory:
    def test_summary_triggered(self):
        llm = DummyLLM(model="mock")
        mem = SummaryMemory(llm, max_messages=4)
        for i in range(6):
            mem.add_message("user", f"msg {i}")
        msgs = mem.get_messages()
        # Should have compressed
        assert mem.summary == "Summarized!"
