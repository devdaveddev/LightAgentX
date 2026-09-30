"""Tests for Lifecycle Hooks — event-driven middleware."""

import pytest

from lightagentx import (
    BaseLLM, LLMResponse, SingleAgent, AgentLoop,
    HookRegistry, HookEvent, tool,
)


class MockLLM(BaseLLM):
    def __init__(self, responses=None, default="Mock response"):
        super().__init__(model="mock-hook-test")
        self._responses = list(responses or [])
        self._default = default
        self._call_count = 0

    def chat(self, messages):
        return self._next()

    def chat_with_tools(self, messages, tools):
        return self._next()

    def _next(self):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, str):
                return LLMResponse(content=resp)
            return resp
        return LLMResponse(content=self._default)


# ── HookRegistry basics ─────────────────────────────────────────────────────

class TestHookRegistry:
    def test_register_and_emit(self):
        registry = HookRegistry()
        events = []

        registry.register("test_event", lambda e: events.append(e))
        registry.emit("test_event", key="value")

        assert len(events) == 1
        assert events[0].name == "test_event"
        assert events[0].data["key"] == "value"

    def test_decorator_registration(self):
        registry = HookRegistry()
        events = []

        @registry.on("my_event")
        def handler(event):
            events.append(event.data)

        registry.emit("my_event", x=42)
        assert events == [{"x": 42}]

    def test_multiple_hooks_same_event(self):
        registry = HookRegistry()
        results = []

        registry.register("ev", lambda e: results.append("a"))
        registry.register("ev", lambda e: results.append("b"))
        registry.emit("ev")

        assert results == ["a", "b"]

    def test_emit_nonexistent_event_is_noop(self):
        registry = HookRegistry()
        registry.emit("does_not_exist")  # should not raise

    def test_hook_error_does_not_crash(self):
        registry = HookRegistry()
        results = []

        registry.register("ev", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        registry.register("ev", lambda e: results.append("survived"))
        registry.emit("ev")

        # The second hook should still fire despite the first raising
        assert results == ["survived"]

    def test_clear_specific_event(self):
        registry = HookRegistry()
        events = []
        registry.register("a", lambda e: events.append("a"))
        registry.register("b", lambda e: events.append("b"))

        registry.clear("a")
        registry.emit("a")
        registry.emit("b")

        assert events == ["b"]

    def test_clear_all(self):
        registry = HookRegistry()
        events = []
        registry.register("a", lambda e: events.append("a"))
        registry.register("b", lambda e: events.append("b"))

        registry.clear()
        registry.emit("a")
        registry.emit("b")

        assert events == []

    def test_registered_events(self):
        registry = HookRegistry()
        registry.register("a", lambda e: None)
        registry.register("b", lambda e: None)

        assert set(registry.registered_events) == {"a", "b"}

    def test_repr(self):
        registry = HookRegistry()
        registry.register("x", lambda e: None)
        assert "HookRegistry" in repr(registry)
        assert "'x'" in repr(registry)


# ── HookEvent ────────────────────────────────────────────────────────────────

class TestHookEvent:
    def test_fields(self):
        event = HookEvent(name="test", data={"k": "v"})
        assert event.name == "test"
        assert event.data == {"k": "v"}
        assert isinstance(event.timestamp, float)


# ── Integration with AgentLoop ───────────────────────────────────────────────

class TestHooksInAgentLoop:
    def test_emits_llm_events(self):
        hooks = HookRegistry()
        events = []
        hooks.register("before_llm_call", lambda e: events.append(("before_llm", e.data["model"])))
        hooks.register("after_llm_call", lambda e: events.append(("after_llm", e.data["model"])))

        llm = MockLLM(default="answer")
        loop = AgentLoop(llm=llm, verbose=False, hooks=hooks)
        loop.run("hello")

        assert ("before_llm", "mock-hook-test") in events
        assert ("after_llm", "mock-hook-test") in events

    def test_emits_iteration_event(self):
        hooks = HookRegistry()
        iterations = []
        hooks.register("on_iteration", lambda e: iterations.append(e.data["iteration"]))

        llm = MockLLM(default="answer")
        loop = AgentLoop(llm=llm, verbose=False, hooks=hooks)
        loop.run("hello")

        assert 1 in iterations

    def test_emits_tool_events(self):
        @tool
        def echo(text: str) -> str:
            """Echo text.

            Args:
                text: Input text.
            """
            return text

        hooks = HookRegistry()
        tool_events = []
        hooks.register("before_tool_call", lambda e: tool_events.append(("before", e.data["tool_name"])))
        hooks.register("after_tool_call", lambda e: tool_events.append(("after", e.data["tool_name"], e.data["result"])))

        llm = MockLLM(responses=[
            LLMResponse(
                content="",
                tool_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}],
            ),
            LLMResponse(content="Final answer"),
        ])
        loop = AgentLoop(llm=llm, tools=[echo], verbose=False, hooks=hooks)
        loop.run("test")

        assert ("before", "echo") in tool_events
        assert ("after", "echo", "hi") in tool_events


# ── Integration with SingleAgent ─────────────────────────────────────────────

class TestHooksInSingleAgent:
    def test_emits_agent_start_end(self):
        hooks = HookRegistry()
        events = []
        hooks.register("on_agent_start", lambda e: events.append(("start", e.data["agent_name"])))
        hooks.register("on_agent_end", lambda e: events.append(("end", e.data["output"])))

        llm = MockLLM(default="response")
        agent = SingleAgent(name="Hooked", llm=llm, hooks=hooks, verbose=False)
        result = agent.run("question")

        assert ("start", "Hooked") in events
        assert ("end", "response") in events
        assert result == "response"

    def test_hooks_survive_reset(self):
        hooks = HookRegistry()
        events = []
        hooks.register("on_agent_start", lambda e: events.append("start"))

        llm = MockLLM(responses=[
            LLMResponse(content="r1"),
            LLMResponse(content="r2"),
        ])
        agent = SingleAgent(name="Bot", llm=llm, hooks=hooks, verbose=False)
        agent.run("q1")
        agent.reset()
        agent.run("q2")

        assert events.count("start") == 2
