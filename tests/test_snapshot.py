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
