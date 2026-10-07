"""Tool-call histories must convert correctly for every provider (and across providers)."""

from types import SimpleNamespace

import pytest

from lightagentx.memory.buffer import BufferMemory


def parallel_tool_history(ids=("call_a", "call_b")):
    m = BufferMemory(100)
    m.add_message("system", "You are helpful.")
    m.add_message("user", "status of ORD-1 and ORD-2?")
    m.add_assistant_tool_calls(content="", tool_calls=[
        {"id": ids[0], "name": "order_status", "arguments": {"order_id": "ORD-1"}},
        {"id": ids[1], "name": "order_status", "arguments": {"order_id": "ORD-2"}},
    ])
    m.add_tool_message(ids[0], "shipped")
    m.add_tool_message(ids[1], "pending")
    m.add_message("assistant", "ORD-1 shipped, ORD-2 pending.")
    m.add_message("user", "thanks")
    return m.get_messages()


class TestAnthropic:
    @pytest.fixture(autouse=True)
    def _sdk(self):
        pytest.importorskip("anthropic")
        from lightagentx.llm.anthropic_llm import AnthropicLLM
        self.llm_cls = AnthropicLLM

    def convert(self, messages):
        _, rest = self.llm_cls._split_system(messages)
        return self.llm_cls._convert_messages(rest)

    def test_parallel_tool_results_share_one_user_message(self):
        conv = self.convert(parallel_tool_history())
        assert [c["role"] for c in conv] == ["user", "assistant", "user", "assistant", "user"]
        results = conv[2]["content"]
        assert [b["tool_use_id"] for b in results] == ["call_a", "call_b"]
        assert all(b["type"] == "tool_result" for b in results)

    def test_every_tool_use_is_answered_in_the_next_message(self):
        conv = self.convert(parallel_tool_history())
        for i, c in enumerate(conv):
            uses = [b["id"] for b in c["content"] if isinstance(c["content"], list) and b.get("type") == "tool_use"]
            if uses:
                answered = [b["tool_use_id"] for b in conv[i + 1]["content"]]
                assert answered == uses

    def test_text_after_results_starts_a_new_message(self):
        conv = self.convert(parallel_tool_history())
        assert conv[-1] == {"role": "user", "content": "thanks"}


class TestGemini:
    @pytest.fixture(autouse=True)
    def _sdk(self):
        pytest.importorskip("google.genai")
        from lightagentx.llm.gemini_llm import GeminiLLM
        self.llm_cls = GeminiLLM

    def test_function_responses_carry_the_called_name(self):
        _, contents = self.llm_cls._convert_messages(parallel_tool_history())
        responses = [p.function_response.name for c in contents for p in c.parts if p.function_response]
        assert responses == ["order_status", "order_status"]

    def test_parallel_responses_share_one_turn(self):
        _, contents = self.llm_cls._convert_messages(parallel_tool_history())
        assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
        assert len(contents[2].parts) == 2

    def test_parallel_calls_get_unique_ids(self, monkeypatch):
        from google.genai import types as gt
        llm = self.llm_cls(api_key="x" * 39)
        parts = [gt.Part.from_function_call(name="order_status", args={"order_id": o})
                 for o in ("ORD-1", "ORD-2")]
        fake = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])
        monkeypatch.setattr(llm._client.models, "generate_content", lambda **kw: fake)
        resp = llm.chat_with_tools([{"role": "user", "content": "hi"}],
                                   [{"type": "function", "function": {"name": "order_status",
                                     "description": "d", "parameters": {"type": "object", "properties": {
                                         "order_id": {"type": "string"}}}}}])
        ids = [tc["id"] for tc in resp.tool_calls]
        assert len(set(ids)) == 2


def test_history_from_gemini_converts_cleanly_for_claude(monkeypatch):
    """Cross-provider: SmartRouter may answer with Gemini, then hand the history to Claude."""
    pytest.importorskip("anthropic")
    pytest.importorskip("google.genai")
    from google.genai import types as gt

    from lightagentx.llm.anthropic_llm import AnthropicLLM
    from lightagentx.llm.gemini_llm import GeminiLLM

    gemini = GeminiLLM(api_key="x" * 39)
    parts = [gt.Part.from_function_call(name="order_status", args={"order_id": o}) for o in ("ORD-1", "ORD-2")]
    fake = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])
    monkeypatch.setattr(gemini._client.models, "generate_content", lambda **kw: fake)
    calls = gemini.chat_with_tools([{"role": "user", "content": "q"}], [{"type": "function", "function": {
        "name": "order_status", "description": "d", "parameters": {"type": "object", "properties": {}}}}]).tool_calls

    history = parallel_tool_history(ids=tuple(c["id"] for c in calls))
    _, rest = AnthropicLLM._split_system(history)
    conv = AnthropicLLM._convert_messages(rest)
    uses = [b["id"] for b in conv[1]["content"] if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for b in conv[2]["content"]]
    assert len(set(uses)) == 2 and results == uses
