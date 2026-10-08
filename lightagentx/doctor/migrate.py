"""Deterministic migrations of saved agents across LightAgentX versions (no LLM)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SUMMARY_MARKER = "CONVERSATION SUMMARY SO FAR:\n"


@dataclass
class MigrationPlan:
    path: Path
    kind: str                    # "snapshot" | "registry"
    description: str
    new_text: str | None = None  # snapshot: the migrated file content
    verified: bool = False
    verification: str = ""


def migrate_snapshot_data(data: dict[str, Any]) -> dict[str, Any]:
    """Old layout -> current: system message out of the transcript, summary and window kept."""
    new = copy.deepcopy(data)
    mem = new.get("memory")
    if not mem:
        return new
    old_messages = mem.get("messages", [])
    if mem.get("type") == "SummaryMemory" and "summary" not in mem:
        summary = ""
        for m in old_messages:
            content = m.get("content") or ""
            if m.get("role") == "system" and _SUMMARY_MARKER in content:
                summary = content.split(_SUMMARY_MARKER, 1)[1].strip()
        mem["summary"] = summary
    mem["messages"] = [m for m in old_messages if m.get("role") != "system"]
    if "max_messages" not in mem:
        # Old snapshots didn't record it; these were the defaults they restored with.
        mem["max_messages"] = 10 if mem.get("type") == "SummaryMemory" else 200
    return new


def plan_snapshot(path: Path) -> MigrationPlan:
    from ..llm.base import BaseLLM, LLMResponse
    from ..snapshot import AgentSnapshot

    old = json.loads(path.read_text(encoding="utf-8"))
    new = migrate_snapshot_data(old)
    plan = MigrationPlan(path, "snapshot", "Rewrite snapshot in the current layout",
                         new_text=json.dumps(new, indent=2, default=str))

    class _NoCalls(BaseLLM):
        def __init__(self):
            super().__init__(model="verify")

        def chat(self, messages):
            raise AssertionError("migration verification must not call an LLM")

        chat_with_tools = lambda self, m, t: self.chat(m)

    # Verify: the migrated file restores, keeps every conversation message, and keeps the summary.
    try:
        agent = AgentSnapshot.from_dict(json.loads(plan.new_text), llm=_NoCalls())
        kept = [m for m in agent.memory.get_messages() if m.get("role") != "system"]
        expected = [m for m in old.get("memory", {}).get("messages", []) if m.get("role") != "system"] \
            if old.get("memory") else []
        problems = []
        if len(kept) != len(expected):
            problems.append(f"{len(expected)} messages before, {len(kept)} after")
        mem = new.get("memory") or {}
        if mem.get("type") == "SummaryMemory" and getattr(agent.memory, "summary", "") != mem.get("summary", ""):
            problems.append("summary not restored")
        if mem and type(agent.memory).__name__ != mem.get("type"):
            problems.append(f"restored as {type(agent.memory).__name__}, expected {mem.get('type')}")
        plan.verified = not problems
        plan.verification = ("restores correctly: "
                             f"{len(kept)} messages, {type(agent.memory).__name__}"
                             + (", summary kept" if mem.get("summary") else "")
                             ) if not problems else "; ".join(problems)
    except Exception as e:  # noqa: BLE001 - report any failure as unverified
        plan.verification = f"migrated file doesn't load: {type(e).__name__}: {e}"
    return plan


def plan_registry(path: Path) -> MigrationPlan:
    from ..state.registry import AgentRegistry
    from ..state.store import REGISTRY_FORMAT, REGISTRY_FORMAT_VERSION, IntegrityError

    plan = MigrationPlan(path, "registry",
                         f"Record format '{REGISTRY_FORMAT} {REGISTRY_FORMAT_VERSION}' after verifying all agents")
    reg = AgentRegistry(path)
    try:
        checked = sum(reg.verify(aid) for aid in reg.store.list_agent_ids())
        plan.verified = True
        plan.verification = f"{len(reg.store.list_agent_ids())} agent(s), {checked} version(s) verified"
    except IntegrityError as e:
        plan.verification = f"integrity check failed: {e}"
    return plan


def apply_migration(plan: MigrationPlan, backup_dir: Path) -> None:
    from ..state.store import REGISTRY_FORMAT, REGISTRY_FORMAT_VERSION, atomic_write

    if not plan.verified:
        raise RuntimeError("Refusing to apply an unverified migration.")
    if plan.kind == "snapshot":
        backup = backup_dir / plan.path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(plan.path.read_bytes())
        atomic_write(plan.path, plan.new_text)
    elif plan.kind == "registry":
        atomic_write(plan.path / "FORMAT", f"{REGISTRY_FORMAT} {REGISTRY_FORMAT_VERSION}\n")
