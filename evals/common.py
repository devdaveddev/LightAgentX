"""Shared helpers for the LightAgentX evaluations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightagentx.llm.base import BaseLLM, LLMResponse  # noqa: E402

RESULTS = Path(__file__).with_name("results")


def add_llm_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--provider", default="scripted",
                    choices=["scripted", "openai", "anthropic", "gemini"],
                    help="'scripted' runs the LLM-free deterministic variant.")
    ap.add_argument("--model", help="Model name for the provider.")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (Ollama, vLLM, ...).")


def make_llm(args: argparse.Namespace) -> BaseLLM:
    kw: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
    if args.model:
        kw["model"] = args.model
    if args.provider == "anthropic":
        from lightagentx.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(**kw)
    if args.provider == "gemini":
        from lightagentx.llm.gemini_llm import GeminiLLM
        return GeminiLLM(**kw)
    from lightagentx.llm.openai_llm import OpenAILLM
    if args.base_url:
        kw["base_url"] = args.base_url
        kw["api_key"] = os.environ.get("OPENAI_API_KEY", "local")
    return OpenAILLM(**kw)


def model_label(args: argparse.Namespace, llm: BaseLLM | None) -> str:
    return "scripted" if args.provider == "scripted" else f"{args.provider}:{llm.model}"


class SpyLLM(BaseLLM):
    """Wraps an LLM and records every message list it is sent."""

    def __init__(self, inner: BaseLLM):
        super().__init__(model=inner.model, temperature=inner.temperature,
                         max_tokens=inner.max_tokens)
        self.inner = inner
        self.inputs: list[list[dict]] = []

    def chat(self, messages):
        self.inputs.append(messages)
        return self.inner.chat(messages)

    def chat_with_tools(self, messages, tools):
        self.inputs.append(messages)
        return self.inner.chat_with_tools(messages, tools)

    def seen_text(self) -> str:
        return json.dumps(self.inputs, default=str)


def _context_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            parts.append(str(tc.get("function", {}).get("arguments", "")))
    return "\n".join(parts)


class PerfectReaderLLM(BaseLLM):
    """
    LLM-free stand-in: answers questions using ONLY what is in its context,
    with perfect recall. Measures whether information *survived* into the
    context — an upper bound for any real model.
    """

    def __init__(self, extract: dict[str, str]):
        super().__init__(model="perfect-reader")
        self.extract = extract  # field -> regex with one group

    def chat(self, messages):
        ctx = _context_text(messages)
        last = messages[-1].get("content") or ""
        if "ORDER=" in last and "ADDRESS=" in last:  # the probe question
            found = {}
            for name, rx in self.extract.items():
                hits = re.findall(rx, ctx)
                found[name] = hits[-1] if hits else "unknown"
            return LLMResponse(content="; ".join(f"{k}={v}" for k, v in found.items()))
        return LLMResponse(content="Noted.")

    def chat_with_tools(self, messages, tools):
        return self.chat(messages)


class WorstCaseLeakerLLM(BaseLLM):
    """
    LLM-free adversary: obeys every extraction attempt perfectly. It calls
    get_state if available, then repeats its whole context verbatim. Its
    leakage equals its exposure — an upper bound for any real model.
    """

    def __init__(self):
        super().__init__(model="worst-case-leaker")

    def chat(self, messages):
        return LLMResponse(content="Everything I know:\n" + _context_text(messages))

    def chat_with_tools(self, messages, tools):
        names = {t["function"]["name"] for t in tools}
        already = any(m.get("role") == "tool" for m in messages[-3:])
        if "get_state" in names and not already:
            return LLMResponse(tool_calls=[{"id": "leak1", "name": "get_state", "arguments": {}}])
        return self.chat(messages)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a 95% Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(successes: int, n: int) -> str:
    p, lo, hi = wilson(successes, n)
    return f"{100 * p:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] (n={n})"
