"""Tests for SmartRouter — auto model switching."""

import pytest

from lightagentx import BaseLLM, LLMResponse, SmartRouter


# ── Mock LLMs ────────────────────────────────────────────────────────────────

class MockModel(BaseLLM):
    """A mock LLM that returns a fixed string."""

    def __init__(self, model: str = "mock-cheap", response: str = "ok"):
        super().__init__(model=model)
        self._response = response
        self.call_count = 0

    def chat(self, messages):
        self.call_count += 1
        return LLMResponse(content=self._response)

    def chat_with_tools(self, messages, tools):
        self.call_count += 1
        return LLMResponse(content=self._response)


class FailingModel(BaseLLM):
    """A mock LLM that always raises."""

    def __init__(self, model: str = "mock-fail"):
        super().__init__(model=model)

    def chat(self, messages):
        raise RuntimeError("Model unavailable")

    def chat_with_tools(self, messages, tools):
        raise RuntimeError("Model unavailable")


# ── Initialization ───────────────────────────────────────────────────────────

class TestSmartRouterInit:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one model"):
            SmartRouter(models=[])

    def test_rejects_unknown_strategy(self):
        m = MockModel()
        with pytest.raises(ValueError, match="Unknown strategy"):
            SmartRouter(models=[m], strategy="magic")

    def test_repr(self):
        m = MockModel(model="gpt-small")
        r = SmartRouter(models=[m], strategy="complexity")
        assert "gpt-small" in repr(r)
        assert "complexity" in repr(r)

    def test_model_property(self):
        m1 = MockModel(model="a")
        m2 = MockModel(model="b")
        r = SmartRouter(models=[m1, m2])
        assert "a" in r.model and "b" in r.model


# ── Complexity Strategy ──────────────────────────────────────────────────────

class TestComplexityStrategy:
    def test_short_message_uses_cheap_model(self):
        cheap = MockModel(model="cheap", response="cheap_answer")
        expensive = MockModel(model="expensive", response="expensive_answer")
        router = SmartRouter(models=[cheap, expensive], strategy="complexity")

        result = router.chat([{"role": "user", "content": "Hi"}])
        assert result.content == "cheap_answer"
        assert cheap.call_count == 1
        assert expensive.call_count == 0

    def test_complex_message_uses_expensive_model(self):
        cheap = MockModel(model="cheap", response="cheap_answer")
        expensive = MockModel(model="expensive", response="expensive_answer")
        router = SmartRouter(
            models=[cheap, expensive],
            strategy="complexity",
            complexity_threshold=0.3,
        )

        long_complex_msg = (
            "Please analyze in detail and compare the trade-offs of "
            "different architecture patterns. Step by step, evaluate "
            "the implications of each approach and provide a comprehensive "
            "reasoning about which design is optimal. " * 5
        )
        result = router.chat([{"role": "user", "content": long_complex_msg}])
        assert result.content == "expensive_answer"
        assert expensive.call_count == 1

    def test_empty_messages_uses_cheap(self):
        cheap = MockModel(model="cheap", response="c")
        expensive = MockModel(model="expensive", response="e")
        router = SmartRouter(models=[cheap, expensive], strategy="complexity")
        result = router.chat([{"role": "system", "content": "sys"}])
        assert result.content == "c"

    def test_conversation_depth_factor(self):
        cheap = MockModel(model="cheap", response="c")
        expensive = MockModel(model="expensive", response="e")
        router = SmartRouter(
            models=[cheap, expensive],
            strategy="complexity",
            complexity_threshold=0.3,
        )
        # 25 messages + complexity keywords should trigger expensive
        messages = [{"role": "user", "content": "analyze this step by step"}] * 25
        result = router.chat(messages)
        assert result.content == "e"

    def test_single_model_always_returns_it(self):
        solo = MockModel(model="solo", response="solo_answer")
        router = SmartRouter(models=[solo], strategy="complexity")
        result = router.chat([{"role": "user", "content": "complex analysis"}])
        assert result.content == "solo_answer"


# ── Round Robin Strategy ─────────────────────────────────────────────────────

class TestRoundRobinStrategy:
    def test_cycles_through_models(self):
        m1 = MockModel(model="a", response="r1")
        m2 = MockModel(model="b", response="r2")
        m3 = MockModel(model="c", response="r3")
        router = SmartRouter(models=[m1, m2, m3], strategy="round_robin")

        assert router.chat([{"role": "user", "content": "q"}]).content == "r1"
        assert router.chat([{"role": "user", "content": "q"}]).content == "r2"
        assert router.chat([{"role": "user", "content": "q"}]).content == "r3"
        assert router.chat([{"role": "user", "content": "q"}]).content == "r1"


# ── Fallback Strategy ────────────────────────────────────────────────────────

class TestFallbackStrategy:
    def test_falls_back_on_error(self):
        failing = FailingModel(model="unreliable")
        backup = MockModel(model="backup", response="backup_answer")
        router = SmartRouter(models=[failing, backup], strategy="fallback")

        result = router.chat([{"role": "user", "content": "hello"}])
        assert result.content == "backup_answer"

    def test_all_fail_raises(self):
        f1 = FailingModel(model="f1")
        f2 = FailingModel(model="f2")
        router = SmartRouter(models=[f1, f2], strategy="fallback")

        with pytest.raises(RuntimeError, match="All .* models failed"):
            router.chat([{"role": "user", "content": "hello"}])

    def test_uses_first_if_working(self):
        primary = MockModel(model="primary", response="primary_answer")
        backup = MockModel(model="backup", response="backup_answer")
        router = SmartRouter(models=[primary, backup], strategy="fallback")

        result = router.chat([{"role": "user", "content": "hello"}])
        assert result.content == "primary_answer"
        assert primary.call_count == 1
        assert backup.call_count == 0


# ── Cost Limit Strategy ──────────────────────────────────────────────────────

class TestCostLimitStrategy:
    def test_upgrades_after_limit(self):
        cheap = MockModel(model="cheap", response="c")
        expensive = MockModel(model="expensive", response="e")
        router = SmartRouter(
            models=[cheap, expensive],
            strategy="cost_limit",
            cost_limit_calls=3,
        )

        for _ in range(3):
            assert router.chat([{"role": "user", "content": "q"}]).content == "c"

        assert router.chat([{"role": "user", "content": "q"}]).content == "e"


# ── Routing History ──────────────────────────────────────────────────────────

class TestRoutingHistory:
    def test_history_recorded(self):
        m = MockModel(model="test-model")
        router = SmartRouter(models=[m], strategy="complexity")
        router.chat([{"role": "user", "content": "hi"}])

        assert len(router.routing_history) == 1
        entry = router.routing_history[0]
        assert entry["model"] == "test-model"
        assert entry["success"] is True
        assert "elapsed_seconds" in entry

    def test_fallback_records_failures(self):
        failing = FailingModel(model="bad")
        good = MockModel(model="good")
        router = SmartRouter(models=[failing, good], strategy="fallback")
        router.chat([{"role": "user", "content": "hi"}])

        assert len(router.routing_history) == 2
        assert router.routing_history[0]["success"] is False
        assert router.routing_history[1]["success"] is True


# ── chat_with_tools passthrough ──────────────────────────────────────────────

class TestChatWithToolsPassthrough:
    def test_routes_tool_calls(self):
        m = MockModel(model="tool-model", response="tool_response")
        router = SmartRouter(models=[m], strategy="complexity")
        result = router.chat_with_tools(
            [{"role": "user", "content": "use a tool"}],
            [{"type": "function", "function": {"name": "test"}}],
        )
        assert result.content == "tool_response"
        assert m.call_count == 1
