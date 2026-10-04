"""SmartOS — a crew of specialist agents that manage the OS through one sandbox."""

from __future__ import annotations

import json
import platform
from typing import Any

from ..agents.base import BaseAgent
from ..agents.single import SingleAgent
from ..hooks import HookRegistry
from ..llm.base import BaseLLM
from ..memory.buffer import BufferMemory
from ..sandbox import Sandbox
from ..tools.base import BaseTool
from ..utils.logger import AgentLogger
from .apps import make_app_tools
from .files import make_file_tools
from .security import make_security_tools
from .shell import make_shell_tools
from .tasks import make_task_tools


_BASE_PROMPT = """You are {role} inside LightX SmartOS, an agent layer that manages the user's {os_name} computer.

{duty}

Sandbox rules (enforced by the system, not by you):
{sandbox}

Guidelines:
- Use your tools to act; don't just describe what the user could do.
- Risky actions automatically ask the user for confirmation — just call the tool.
- If a tool returns an error mentioning the sandbox, policy, or that the user declined,
  do NOT work around it with other commands, paths or tools. Explain and stop.
- Relative paths are inside the workspace; use ~ for the user's home folder.
- Reply concisely, in plain language. Summarize long tool output instead of pasting it."""


_ROUTER_PROMPT = """You route requests inside an agent-managed operating system.

Specialists:
{agents}

Recent conversation:
{history}

Decide who handles the user's NEW message. Respond ONLY with a JSON array:
[{{"agent": "<name>", "task": "<self-contained task>"}}]

Rules:
- Use exactly one specialist unless the request truly needs several.
- Make each task self-contained: resolve words like "it", "that one", "the second"
  using the conversation, and include concrete PIDs, paths and names.
- For greetings or questions about yourself, use "Generalist"."""


_SPECIALISTS: dict[str, tuple[str, str]] = {
    "FileManager": (
        "Finds, reads, writes, organizes, opens and deletes files and folders.",
        "You manage files and folders: browse, search, read, write, open them in apps, "
        "and delete (to a recoverable trash).",
    ),
    "TaskManager": (
        "Task manager: system load, CPU/RAM/disk, running processes, stopping processes.",
        "You are the task manager: report system resources, find processes using CPU or "
        "memory, and terminate processes the user wants stopped. Always identify the exact "
        "PID with list_processes before terminating.",
    ),
    "AppManager": (
        "Finds installed applications, launches them, closes them, opens files in apps.",
        "You manage desktop applications: search installed apps, launch them, close them, "
        "and open files with their default app.",
    ),
    "SecurityGuard": (
        "Security: scans firewall, open ports, network connections, suspicious processes, "
        "autostart persistence, credential file permissions; handles security issues.",
        "You are the security analyst. Investigate with read-only checks first, explain "
        "findings with severity (info/warning/critical), and recommend fixes. Only terminate "
        "processes when the evidence is clear.",
    ),
    "Operator": (
        "Runs shell commands and Python code in the sandbox for anything else "
        "(calculations, text processing, troubleshooting commands).",
        "You run shell commands and Python in the sandbox to get things done. The sandbox "
        "filesystem is read-only except the workspace, and has no network unless enabled.",
    ),
}


def build_os_tools(sandbox: Sandbox) -> dict[str, list[BaseTool]]:
    """Build every SmartOS tool and group them by specialist."""
    files = {t.name: t for t in make_file_tools(sandbox)}
    tasks = {t.name: t for t in make_task_tools(sandbox)}
    apps = {t.name: t for t in make_app_tools(sandbox)}
    sec = {t.name: t for t in make_security_tools(sandbox)}
    shell = {t.name: t for t in make_shell_tools(sandbox)}

    return {
        "FileManager": list(files.values()),
        "TaskManager": list(tasks.values()),
        "AppManager": list(apps.values()) + [files["open_file"], files["search_files"]],
        "SecurityGuard": list(sec.values()) + [
            tasks["list_processes"], tasks["process_details"], tasks["terminate_process"],
            files["file_info"],
        ],
        "Operator": list(shell.values()) + [
            files["list_directory"], files["read_file"], files["write_file"],
        ],
    }


class SmartOS(BaseAgent):
    """
    Chat with your computer. A router LLM sends each message to the right
    specialist (files, tasks, apps, security, shell); every specialist acts
    only through the shared Sandbox.

    Usage::

        from lightagentx import OpenAILLM
        from lightagentx.sandbox import Sandbox
        from lightagentx.smartos import SmartOS

        os_agent = SmartOS(llm=OpenAILLM(), sandbox=Sandbox())
        print(os_agent.run("what's eating my RAM?"))

    Args:
        llm: LLM used by the router and all specialists.
        sandbox: The security sandbox. Defaults to a deny-by-default Sandbox
            (risky actions are refused unless you attach a confirmer).
        mode: "crew" (router + specialists) or "single" (one agent, all tools —
            fewer LLM calls, good for small/cheap models).
        hooks: Shared lifecycle hooks (e.g. to display tool calls).
    """

    def __init__(
        self,
        llm: BaseLLM,
        sandbox: Sandbox | None = None,
        mode: str = "crew",
        hooks: HookRegistry | None = None,
        verbose: bool = False,
        max_iterations: int = 12,
        history_turns: int = 6,
    ):
        super().__init__(name="SmartOS", description="Agent-managed operating system")
        if mode not in ("crew", "single"):
            raise ValueError("mode must be 'crew' or 'single'")
        self.llm = llm
        self.sandbox = sandbox or Sandbox()
        self.mode = mode
        self.hooks = hooks or HookRegistry()
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.history_turns = history_turns
        self.logger = AgentLogger(verbose=verbose)
        self._history: list[tuple[str, str]] = []

        grouped = build_os_tools(self.sandbox)
        all_tools: dict[str, BaseTool] = {}
        for tools in grouped.values():
            for t in tools:
                all_tools.setdefault(t.name, t)

        self.generalist = self._make_agent(
            "Generalist", "Handles anything; has every tool.",
            "You are the general OS assistant with access to every tool.",
            list(all_tools.values()),
        )
        self.specialists: dict[str, SingleAgent] = {}
        if mode == "crew":
            for name, (desc, duty) in _SPECIALISTS.items():
                self.specialists[name] = self._make_agent(name, desc, duty, grouped[name])

    def _make_agent(self, name: str, description: str, duty: str,
                    tools: list[BaseTool]) -> SingleAgent:
        prompt = _BASE_PROMPT.format(
            role=f"the {name} agent", os_name=f"{platform.system()} {platform.release()}",
            duty=duty, sandbox=self.sandbox.describe(),
        )
        return SingleAgent(
            name=name, llm=self.llm, tools=tools, memory=BufferMemory(max_messages=40),
            system_prompt=prompt, description=description,
            max_iterations=self.max_iterations, verbose=self.verbose, hooks=self.hooks,
        )

    # ── Routing ───────────────────────────────────────────────────────────

    def route(self, message: str) -> list[dict[str, str]]:
        """Ask the router LLM which specialist(s) should handle a message."""
        if self.mode == "single":
            return [{"agent": "Generalist", "task": message}]

        agents = "\n".join(f"- {n}: {a.description}" for n, a in self.specialists.items())
        agents += "\n- Generalist: greetings, questions about SmartOS, or mixed requests."
        history = "\n".join(
            f"User: {u}\nSmartOS: {a[:400]}" for u, a in self._history[-self.history_turns:]
        ) or "(none)"

        response = self.llm.chat([
            {"role": "system", "content": _ROUTER_PROMPT.format(agents=agents, history=history)},
            {"role": "user", "content": message},
        ])
        plan = _parse_plan(response.content)
        valid = [d for d in plan
                 if d.get("agent") in self.specialists or d.get("agent") == "Generalist"]
        if not valid:
            self.logger.error("Router gave no usable plan; using Generalist")
            return [{"agent": "Generalist", "task": message}]
        return valid

    def _agent(self, name: str) -> SingleAgent:
        return self.specialists.get(name, self.generalist)

    # ── Running ───────────────────────────────────────────────────────────

    def run(self, input_text: str) -> str:
        self.hooks.emit("on_agent_start", agent_name=self.name, input_text=input_text)
        plan = self.route(input_text)
        self.hooks.emit("on_route", plan=plan)

        if len(plan) == 1:
            d = plan[0]
            answer = self._agent(d["agent"]).run(d.get("task") or input_text)
        else:
            parts = []
            for d in plan:
                result = self._agent(d["agent"]).run(d.get("task") or input_text)
                parts.append(f"### {d['agent']}\n{result}")
            answer = self.llm.chat([{
                "role": "user",
                "content": (f"Combine these specialist reports into one concise answer to the "
                            f"user's request: {input_text}\n\n" + "\n\n".join(parts)),
            }]).content

        self._history.append((input_text, answer))
        self._history = self._history[-self.history_turns:]
        self.hooks.emit("on_agent_end", agent_name=self.name, input_text=input_text, output=answer)
        return answer

    def reset(self) -> None:
        """Forget the conversation in every agent."""
        self._history.clear()
        for agent in [self.generalist, *self.specialists.values()]:
            agent.reset()


def _parse_plan(content: str) -> list[dict[str, Any]]:
    text = (content or "").strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
