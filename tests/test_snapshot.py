"""Tests for Agent Snapshots — portable stateful agent export/import."""

import json
import pytest
from pathlib import Path

from lightagentx import (
    BaseLLM, LLMResponse, SingleAgent, AgentSnapshot,
    BufferMemory, tool,
)


class MockLLM(BaseLLM):
    def __init__(self, model="mock", response="ok"):
        super().__init__(model=model)
        self._response = response

    def chat(self, messages):
        return LLMResponse(content=self._response)

    def chat_with_tools(self, messages, tools):
        return LLMResponse(content=self._response)


@tool
def mock_tool(text: str) -> str:
    """Echo text back.

    Args:
        text: The text to echo.
    """
    return text


# ── to_dict / from_dict ──────────────────────────────────────────────────────

class TestToDict:
    def test_captures_agent_config(self):
        llm = MockLLM(model="gpt-test")
        agent = SingleAgent(
            name="TestBot",
            llm=llm,
            system_prompt="Be helpful.",
            description="A test agent",
            max_iterations=5,
            verbose=False,
        )
        data = AgentSnapshot.to_dict(agent)

        assert data["agent"]["name"] == "TestBot"
        assert data["agent"]["system_prompt"] == "Be helpful."
        assert data["agent"]["description"] == "A test agent"
        assert data["agent"]["max_iterations"] == 5
        assert data["format_version"] == 1

    def test_captures_llm_config(self):
        llm = MockLLM(model="gpt-custom")
        llm.temperature = 0.5
        llm.max_tokens = 2048
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        data = AgentSnapshot.to_dict(agent)

        assert data["llm_config"]["model"] == "gpt-custom"
        assert data["llm_config"]["temperature"] == 0.5
        assert data["llm_config"]["max_tokens"] == 2048

    def test_captures_tool_manifest(self):
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, tools=[mock_tool], verbose=False)
        data = AgentSnapshot.to_dict(agent)

        assert len(data["tool_manifest"]) == 1
        assert data["tool_manifest"][0]["name"] == "mock_tool"
        assert "parameters" in data["tool_manifest"][0]

    def test_captures_memory(self):
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        agent.run("Hello")  # adds messages to memory
        data = AgentSnapshot.to_dict(agent)

        assert data["memory"] is not None
        assert len(data["memory"]["messages"]) > 0

    def test_without_memory(self):
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        data = AgentSnapshot.to_dict(agent, include_memory=False)
        assert data["memory"] is None

    def test_rejects_non_single_agent(self):
        from lightagentx import BaseAgent

        class CustomAgent(BaseAgent):
            def run(self, input_text):
                return "x"

        with pytest.raises(TypeError, match="SingleAgent only"):
            AgentSnapshot.to_dict(CustomAgent(name="Bad"))


class TestFromDict:
    def test_restores_agent_config(self):
        llm = MockLLM()
        agent = SingleAgent(
            name="Original",
            llm=llm,
            system_prompt="Custom prompt",
            description="Test desc",
            max_iterations=3,
            verbose=False,
        )
        data = AgentSnapshot.to_dict(agent)

        restored = AgentSnapshot.from_dict(data, llm=MockLLM())
        assert restored.name == "Original"
        assert restored.system_prompt == "Custom prompt"
        assert restored.description == "Test desc"
        assert restored.max_iterations == 3

    def test_restores_memory(self):
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        agent.run("First question")
        data = AgentSnapshot.to_dict(agent)

        restored = AgentSnapshot.from_dict(data, llm=MockLLM())
        messages = restored.memory.get_messages()
        assert len(messages) > 0

    def test_warns_on_missing_tools(self):
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, tools=[mock_tool], verbose=False)
        data = AgentSnapshot.to_dict(agent)

        with pytest.warns(UserWarning, match="mock_tool"):
            AgentSnapshot.from_dict(data, llm=MockLLM(), tools=[])

    def test_rejects_bad_format_version(self):
        data = {"format_version": 999, "agent": {}, "llm_config": {}}
        with pytest.raises(ValueError, match="not compatible"):
            AgentSnapshot.from_dict(data, llm=MockLLM())


# ── Save / Load File Round Trip ──────────────────────────────────────────────

class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "test_agent.agent.json"
        llm = MockLLM(model="save-test")
        agent = SingleAgent(
            name="Saver",
            llm=llm,
            system_prompt="Save me.",
            verbose=False,
        )
        agent.run("test input")

        AgentSnapshot.save(agent, path)
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["agent"]["name"] == "Saver"

        restored = AgentSnapshot.load(path, llm=MockLLM())
        assert restored.name == "Saver"
        assert restored.system_prompt == "Save me."

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "agent.json"
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        AgentSnapshot.save(agent, path)
        assert path.exists()


# ── Portable Export ──────────────────────────────────────────────────────────

class TestExportPortable:
    def test_no_memory_in_portable(self, tmp_path):
        path = tmp_path / "card.json"
        llm = MockLLM()
        agent = SingleAgent(name="Bot", llm=llm, verbose=False)
        agent.run("secret conversation")

        AgentSnapshot.export_portable(agent, path)
        data = json.loads(path.read_text())
        assert data["memory"] is None


# ── SingleAgent convenience methods ──────────────────────────────────────────

class TestSingleAgentConvenience:
    def test_snapshot_method(self, tmp_path):
        path = tmp_path / "conv.agent.json"
        llm = MockLLM()
        agent = SingleAgent(name="Conv", llm=llm, verbose=False)
        agent.snapshot(path)
        assert path.exists()

    def test_from_snapshot_classmethod(self, tmp_path):
        path = tmp_path / "load.agent.json"
        llm = MockLLM()
        agent = SingleAgent(name="Orig", llm=llm, verbose=False)
        agent.snapshot(path)

        restored = SingleAgent.from_snapshot(path, llm=MockLLM())
        assert restored.name == "Orig"


# ── Memory fidelity ─────────────────────────────────────────────────────────

from lightagentx import SummaryMemory  # noqa: E402


class CountingSummarizer(BaseLLM):
    """Summarizer LLM whose output is distinct from any agent reply."""

    def __init__(self):
        super().__init__(model="summarizer")
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return LLMResponse(content=f"SUMMARY-{self.calls}")

    def chat_with_tools(self, messages, tools):
        return self.chat(messages)


def summary_agent(summarizer):
    agent = SingleAgent(name="Bot", llm=MockLLM(response="reply"),
                        memory=SummaryMemory(llm=summarizer, max_messages=4), verbose=False)
    for q in ["one", "two", "three"]:
        agent.run(q)
    return agent


class TestMemoryFidelity:
    def test_summary_memory_round_trip(self, tmp_path):
        summarizer = CountingSummarizer()
        agent = summary_agent(summarizer)
        assert agent.memory.summary  # compression happened
        agent.snapshot(tmp_path / "a.json")

        calls_before = summarizer.calls
        restored = SingleAgent.from_snapshot(tmp_path / "a.json", llm=summarizer)

        assert isinstance(restored.memory, SummaryMemory)
        assert restored.memory.max_messages == 4
        assert restored.memory.summary == agent.memory.summary
        assert restored.memory.get_messages() == agent.memory.get_messages()
        assert summarizer.calls == calls_before  # restoring never re-summarizes

    def test_restored_summary_memory_keeps_compressing(self, tmp_path):
        summarizer = CountingSummarizer()
        summary_agent(summarizer).snapshot(tmp_path / "a.json")
        restored = SingleAgent.from_snapshot(tmp_path / "a.json", llm=summarizer)
        old = restored.memory.summary
        for q in ["four", "five", "six"]:
            restored.run(q)
        assert restored.memory.summary != old

    def test_system_prompt_not_duplicated_in_file(self, tmp_path):
        agent = SingleAgent(name="Bot", llm=MockLLM(), system_prompt="Be terse.", verbose=False)
        agent.run("hi")
        data = AgentSnapshot.to_dict(agent)
        assert all(m["role"] != "system" for m in data["memory"]["messages"])
        assert data["agent"]["system_prompt"] == "Be terse."

    def test_buffer_memory_window_and_tool_calls_preserved(self, tmp_path):
        class ToolOnce(MockLLM):
            n = 0
            def chat_with_tools(self, messages, tools):
                ToolOnce.n += 1
                if ToolOnce.n == 1:
                    return LLMResponse(tool_calls=[{"id": "c1", "name": "mock_tool",
                                                    "arguments": {"text": "x"}}])
                return LLMResponse(content="done")
        agent = SingleAgent(name="Bot", llm=ToolOnce(), tools=[mock_tool],
                            memory=BufferMemory(max_messages=37), verbose=False)
        agent.run("use the tool")
        agent.snapshot(tmp_path / "a.json")
        restored = SingleAgent.from_snapshot(tmp_path / "a.json", llm=MockLLM(), tools=[mock_tool])
        assert restored.memory.max_messages == 37
        assert restored.memory.get_messages() == agent.memory.get_messages()

    def test_legacy_snapshot_summary_is_recovered(self, tmp_path):
        """Files written before this fix kept the summary inside the system message."""
        agent = SingleAgent(name="Bot", llm=MockLLM(), verbose=False)
        data = AgentSnapshot.to_dict(agent)
        data["memory"] = {"type": "SummaryMemory", "messages": [
            {"role": "system", "content": "You are helpful.\n\nCONVERSATION SUMMARY SO FAR:\nOLD SUMMARY"},
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
        ]}
        restored = AgentSnapshot.from_dict(data, llm=CountingSummarizer())
        assert isinstance(restored.memory, SummaryMemory)
        assert restored.memory.summary == "OLD SUMMARY"
        assert [m["content"] for m in restored.memory.get_messages()[1:]] == ["q", "a"]

    def test_unknown_memory_type_falls_back_with_warning(self):
        agent = SingleAgent(name="Bot", llm=MockLLM(), verbose=False)
        agent.run("hi")
        data = AgentSnapshot.to_dict(agent)
        data["memory"]["type"] = "VectorMemory"
        with pytest.warns(UserWarning, match="VectorMemory"):
            restored = AgentSnapshot.from_dict(data, llm=MockLLM())
        assert isinstance(restored.memory, BufferMemory)
