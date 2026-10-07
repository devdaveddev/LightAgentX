"""Agent Snapshots — serialize and restore agents with full state for portability."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .agents.base import BaseAgent
from .memory.base import BaseMemory
from .memory.buffer import BufferMemory
from .memory.summary import SummaryMemory

_SUMMARY_MARKER = "CONVERSATION SUMMARY SO FAR:\n"


_SNAPSHOT_FORMAT_VERSION = 1


class AgentSnapshot:
    """
    Save and load agent state as portable JSON files.

    This lets you build an agent on one project, export it with its full
    conversation history and configuration, and import it into another
    project — keeping all context intact.

    Usage::

        # Save
        AgentSnapshot.save(agent, "my_agent.agent.json")

        # Load
        restored = AgentSnapshot.load("my_agent.agent.json", llm=llm, tools=[...])
        restored.run("Continue where we left off...")

        # Portable agent card (config only, no memory)
        AgentSnapshot.export_portable(agent, "my_agent_card.json")
    """

    @staticmethod
    def to_dict(agent: BaseAgent, include_memory: bool = True) -> dict[str, Any]:
        """
        Serialize an agent's state to a dictionary.

        Args:
            agent: The agent to serialize (must be a SingleAgent).
            include_memory: If True, include conversation history.

        Returns:
            A JSON-serializable dictionary of the agent's full state.
        """
        from .agents.single import SingleAgent

        if not isinstance(agent, SingleAgent):
            raise TypeError(
                f"Snapshots currently support SingleAgent only, got {type(agent).__name__}"
            )

        snapshot: dict[str, Any] = {
            "format_version": _SNAPSHOT_FORMAT_VERSION,
            "timestamp": time.time(),
            "agent": {
                "name": agent.name,
                "description": agent.description,
                "system_prompt": agent.system_prompt,
                "max_iterations": agent.max_iterations,
                "verbose": agent.verbose,
            },
            "llm_config": {
                "model": agent.llm.model,
                "temperature": agent.llm.temperature,
                "max_tokens": agent.llm.max_tokens,
            },
            "tool_manifest": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in agent.tools
            ],
        }

        if include_memory:
            snapshot["memory"] = AgentSnapshot._memory_to_dict(agent.memory)
        else:
            snapshot["memory"] = None

        return snapshot

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        llm: Any,
        tools: list[Any] | None = None,
    ) -> Any:
        """
        Reconstruct an agent from a snapshot dictionary.

        Args:
            data: Snapshot dict (as returned by ``to_dict``).
            llm: A BaseLLM instance to power the restored agent.
            tools: Tool instances to attach. Tool names should match
                   those in the snapshot's tool_manifest.

        Returns:
            A fully functional SingleAgent with restored state.
        """
        from .agents.single import SingleAgent

        fmt = data.get("format_version", 0)
        if fmt != _SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is not compatible with "
                f"current version {_SNAPSHOT_FORMAT_VERSION}."
            )

        agent_cfg = data["agent"]
        tools = tools or []

        # Validate tools against manifest
        manifest_names = {t["name"] for t in data.get("tool_manifest", [])}
        provided_names = {t.name for t in tools}
        missing = manifest_names - provided_names
        if missing:
            import warnings
            warnings.warn(
                f"Snapshot expects tools {missing} but they were not provided. "
                f"The agent may not work correctly without them.",
                UserWarning,
                stacklevel=2,
            )

        memory = AgentSnapshot._memory_from_dict(data.get("memory"), llm)

        agent = SingleAgent(
            name=agent_cfg["name"],
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=agent_cfg.get("system_prompt", "You are a helpful AI assistant."),
            description=agent_cfg.get("description", ""),
            max_iterations=agent_cfg.get("max_iterations", 10),
            verbose=agent_cfg.get("verbose", True),
        )

        return agent

    # ── memory (de)serialization ──────────────────────────────────────────

    @staticmethod
    def _memory_to_dict(memory: BaseMemory) -> dict[str, Any]:
        """
        Store the conversation WITHOUT the system message (the system prompt
        already lives in the agent config, and a second copy would be
        overwritten on restore). SummaryMemory's running summary is stored
        as its own field so it survives the round trip.
        """
        data: dict[str, Any] = {
            "type": type(memory).__name__,
            "max_messages": getattr(memory, "max_messages", None),
            "messages": [m for m in memory.get_messages() if m.get("role") != "system"],
        }
        if isinstance(memory, SummaryMemory):
            data["summary"] = memory.summary
        return data

    @staticmethod
    def _memory_from_dict(mem_data: dict[str, Any] | None, llm: Any) -> BaseMemory | None:
        """Rebuild the same memory class with the exact stored messages."""
        if not mem_data:
            return None
        messages = [dict(m) for m in mem_data.get("messages", []) if m.get("role") != "system"]
        mem_type = mem_data.get("type", "BufferMemory")

        if mem_type == "SummaryMemory":
            memory: BaseMemory = SummaryMemory(llm=llm, max_messages=mem_data.get("max_messages") or 10)
            summary = mem_data.get("summary")
            if summary is None:  # snapshots written before the summary had its own field
                summary = _legacy_summary(mem_data.get("messages", []))
            memory._summary = summary or ""
        else:
            if mem_type != "BufferMemory":
                import warnings
                warnings.warn(f"Memory type '{mem_type}' is restored as BufferMemory.",
                              UserWarning, stacklevel=3)
            memory = BufferMemory(max_messages=mem_data.get("max_messages") or 200)

        # Messages are already in the stored wire format; load them as-is so the
        # round trip is exact and no summarization is triggered while restoring.
        memory._messages = messages
        return memory

    @staticmethod
    def save(agent: BaseAgent, path: str | Path) -> None:
        """
        Save an agent's full state to a JSON file.

        Args:
            agent: The agent to save.
            path: File path to write to (typically ``*.agent.json``).
        """
        data = AgentSnapshot.to_dict(agent, include_memory=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load(
        path: str | Path,
        llm: Any,
        tools: list[Any] | None = None,
    ) -> Any:
        """
        Load an agent from a snapshot file.

        Args:
            path: Path to the ``.agent.json`` file.
            llm: A BaseLLM instance to power the restored agent.
            tools: Tool instances to attach.

        Returns:
            A fully functional SingleAgent with restored state.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentSnapshot.from_dict(data, llm=llm, tools=tools)

    @staticmethod
    def export_portable(agent: BaseAgent, path: str | Path) -> None:
        """
        Export a minimal agent card — config and tool manifest only, no memory.

        This is useful for sharing agent personas across teams or projects
        without exposing conversation history.

        Args:
            agent: The agent to export.
            path: File path to write to.
        """
        data = AgentSnapshot.to_dict(agent, include_memory=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _legacy_summary(messages: list[dict[str, Any]]) -> str:
    """Recover a SummaryMemory summary from an older snapshot's system message."""
    for m in messages:
        content = m.get("content") or ""
        if m.get("role") == "system" and _SUMMARY_MARKER in content:
            return content.split(_SUMMARY_MARKER, 1)[1].strip()
    return ""
