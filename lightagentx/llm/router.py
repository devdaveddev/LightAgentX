"""SmartRouter — auto-selects the best LLM from a pool based on task characteristics."""

from __future__ import annotations

import time
from typing import Any

from .base import BaseLLM, LLMResponse


_COMPLEXITY_KEYWORDS = frozenset({
    "analyze", "compare", "contrast", "evaluate", "explain why",
    "step by step", "reasoning", "implications", "trade-offs",
    "architecture", "design", "refactor", "debug", "optimize",
    "comprehensive", "in-depth", "detailed", "multi-step",
})


class SmartRouter(BaseLLM):
    """
    A meta-LLM that routes requests to the best model from a pool.

    Models are ordered cheapest-first. The router picks which model handles
    each request based on the chosen strategy.

    Strategies:
        - ``"complexity"`` — heuristic analysis of message length, keyword
          markers, and conversation depth to pick cheap vs. powerful models.
        - ``"round_robin"`` — cycles through models evenly.
        - ``"fallback"`` — tries the cheapest model first; on error, falls
          back to the next one.
        - ``"cost_limit"`` — stays on the cheapest model until a token/call
          budget is hit, then upgrades.

    Usage::

        router = SmartRouter(
            models=[
                OpenAILLM(model="gpt-4o-mini"),
                OpenAILLM(model="gpt-4o"),
            ],
            strategy="complexity",
        )
        agent = SingleAgent(name="Bot", llm=router)
    """

    def __init__(
        self,
        models: list[BaseLLM],
        strategy: str = "complexity",
        complexity_threshold: float = 0.5,
        cost_limit_calls: int = 20,
    ):
        if not models:
            raise ValueError("SmartRouter requires at least one model.")
        if strategy not in ("complexity", "round_robin", "fallback", "cost_limit"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: complexity, round_robin, fallback, cost_limit."
            )

        super().__init__(model=f"router({', '.join(m.model for m in models)})")
        self.models = models
        self.strategy = strategy
        self.complexity_threshold = complexity_threshold
        self.cost_limit_calls = cost_limit_calls

        self._rr_index = 0
        self._call_count = 0
        self.routing_history: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Route a chat request to the selected model."""
        selected = self._select_model(messages)
        return self._execute(selected, "chat", messages)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Route a tool-calling request to the selected model."""
        selected = self._select_model(messages)
        return self._execute(selected, "chat_with_tools", messages, tools)

    def _execute(
        self,
        selected: BaseLLM,
        method: str,
        messages: list,
        tools: list | None = None,
    ) -> LLMResponse:
        """Execute the call, with fallback logic if strategy is 'fallback'."""
        start = time.monotonic()

        if self.strategy == "fallback":
            return self._execute_with_fallback(method, messages, tools, start)

        response = self._call_model(selected, method, messages, tools)
        self._record(selected, time.monotonic() - start, success=True)
        return response

    def _execute_with_fallback(
        self,
        method: str,
        messages: list,
        tools: list | None,
        start: float,
    ) -> LLMResponse:
        last_err: Exception | None = None
        for model in self.models:
            try:
                response = self._call_model(model, method, messages, tools)
                self._record(model, time.monotonic() - start, success=True)
                return response
            except Exception as exc:
                last_err = exc
                self._record(
                    model, time.monotonic() - start,
                    success=False, error=str(exc),
                )
                continue

        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_err}"
        )

    @staticmethod
    def _call_model(
        model: BaseLLM,
        method: str,
        messages: list,
        tools: list | None,
    ) -> LLMResponse:
        if method == "chat_with_tools" and tools is not None:
            return model.chat_with_tools(messages, tools)
        return model.chat(messages)

    def _select_model(self, messages: list[dict]) -> BaseLLM:
        """Pick a model based on the active strategy."""
        self._call_count += 1

        if self.strategy == "complexity":
            score = self._classify_complexity(messages)
            idx = self._score_to_index(score)
            return self.models[idx]

        if self.strategy == "round_robin":
            model = self.models[self._rr_index % len(self.models)]
            self._rr_index += 1
            return model

        if self.strategy == "cost_limit":
            if self._call_count <= self.cost_limit_calls:
                return self.models[0]
            return self.models[-1]

        # fallback — start from cheapest (handled in _execute)
        return self.models[0]

    def _classify_complexity(self, messages: list[dict]) -> float:
        """
        Return a complexity score between 0.0 (trivial) and 1.0 (complex).

        Heuristics used:
        - Total character length of the latest user message
        - Presence of complexity-signaling keywords
        - Conversation depth (number of messages)
        """
        score = 0.0

        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return 0.0

        latest = user_messages[-1].get("content", "")

        # Length factor (0-0.3): longer messages tend to be more complex
        char_len = len(latest)
        if char_len > 500:
            score += 0.3
        elif char_len > 200:
            score += 0.2
        elif char_len > 50:
            score += 0.1

        # Keyword factor (0-0.4): presence of complexity markers
        lower_text = latest.lower()
        keyword_hits = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in lower_text)
        score += min(keyword_hits * 0.1, 0.4)

        # Depth factor (0-0.3): longer conversations need more context handling
        depth = len(messages)
        if depth > 20:
            score += 0.3
        elif depth > 10:
            score += 0.2
        elif depth > 4:
            score += 0.1

        return min(score, 1.0)

    def _score_to_index(self, score: float) -> int:
        """Map a 0.0–1.0 complexity score to a model index."""
        n = len(self.models)
        if n == 1:
            return 0
        if score < self.complexity_threshold:
            return 0  # cheapest
        # Linearly map the above-threshold range across remaining models
        above = (score - self.complexity_threshold) / (1.0 - self.complexity_threshold)
        idx = int(above * (n - 1)) + 1
        return min(idx, n - 1)

    def _record(
        self,
        model: BaseLLM,
        elapsed: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "call_number": self._call_count,
            "model": model.model,
            "strategy": self.strategy,
            "elapsed_seconds": round(elapsed, 4),
            "success": success,
        }
        if error:
            entry["error"] = error
        self.routing_history.append(entry)

    def __repr__(self) -> str:
        models_str = ", ".join(m.model for m in self.models)
        return f"SmartRouter(strategy={self.strategy!r}, models=[{models_str}])"
