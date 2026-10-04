"""Tests for SmartOS tools and the routing agent (no API keys needed)."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("psutil")

from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.sandbox import Risk, Sandbox, SandboxPolicy, SandboxViolation, SubprocessBackend, allow_all, deny_all
from lightagentx.smartos import (
    SmartOS,
    make_app_tools,
    make_file_tools,
    make_security_tools,
    make_task_tools,
)
from lightagentx.smartos.apps import discover_apps, find_app
from lightagentx.smartos.tasks import terminate_pid


def tools_by_name(tools):
    return {t.name: t for t in tools}


@pytest.fixture
def sandbox(tmp_path):
    policy = SandboxPolicy(workspace=tmp_path / "ws", read_paths=[tmp_path])
    return Sandbox(policy=policy, backend=SubprocessBackend(), confirmer=allow_all)


class TestFileTools:
    def test_write_read_list(self, sandbox):
        t = tools_by_name(make_file_tools(sandbox))
        t["write_file"](path="notes/todo.txt", content="buy milk")
        assert t["read_file"](path="notes/todo.txt") == "buy milk"
        assert "todo.txt" in t["list_directory"](path="notes")

    def test_read_outside_policy_blocked(self, sandbox):
        t = tools_by_name(make_file_tools(sandbox))
        with pytest.raises(SandboxViolation):
            t["read_file"](path="/etc/hostname")

    def test_secret_files_hidden_from_listing(self, sandbox):
        t = tools_by_name(make_file_tools(sandbox))
        (sandbox.policy.workspace / "server.pem").write_text("x")
        (sandbox.policy.workspace / "ok.txt").write_text("x")
        out = t["list_directory"](path=".")
        assert "ok.txt" in out and "server.pem" not in out

    def test_search(self, sandbox):
        t = tools_by_name(make_file_tools(sandbox))
        (sandbox.policy.workspace / "a").mkdir()
        (sandbox.policy.workspace / "a" / "report.pdf").write_text("x")
        assert "report.pdf" in t["search_files"](pattern="*.pdf", root=".")

    def test_trash_is_recoverable_and_confirmed(self, sandbox, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        t = tools_by_name(make_file_tools(sandbox))
        f = sandbox.policy.workspace / "old.txt"
        f.write_text("bye")
        out = t["move_to_trash"](path="old.txt")
        assert not f.exists() and "trash" in out

    def test_delete_declined(self, sandbox):
        sandbox.confirmer = deny_all
        t = tools_by_name(make_file_tools(sandbox))
        (sandbox.policy.workspace / "keep.txt").write_text("x")
        with pytest.raises(SandboxViolation, match="declined"):
            t["move_to_trash"](path="keep.txt")
        assert (sandbox.policy.workspace / "keep.txt").exists()

    def test_cannot_delete_workspace_root(self, sandbox):
        t = tools_by_name(make_file_tools(sandbox))
        with pytest.raises(SandboxViolation):
            t["move_to_trash"](path=".")

    def test_opening_launcher_is_high_risk(self, sandbox, monkeypatch):
        seen = []
        sandbox.confirmer = lambda d, r: seen.append(r) or False
        (sandbox.policy.workspace / "evil.desktop").write_text("[Desktop Entry]")
        t = tools_by_name(make_file_tools(sandbox))
        with pytest.raises(SandboxViolation):
            t["open_file"](path="evil.desktop")
        assert seen == [Risk.HIGH]


class TestTaskTools:
    def test_overview_and_list(self, sandbox):
        t = tools_by_name(make_task_tools(sandbox))
        assert "Memory:" in t["system_overview"]()
        assert "PID" in t["list_processes"](limit=5)

    def test_terminate_child_process(self, sandbox):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            out = terminate_pid(sandbox, proc.pid)
            assert "terminated" in out or "exited" in out
        finally:
            proc.kill()

    def test_terminate_requires_confirmation(self, sandbox):
        sandbox.confirmer = deny_all
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with pytest.raises(SandboxViolation, match="declined"):
                terminate_pid(sandbox, proc.pid)
            assert proc.poll() is None
        finally:
            proc.kill()

    def test_cannot_kill_self_or_init(self, sandbox):
        with pytest.raises(SandboxViolation):
            terminate_pid(sandbox, os.getpid())
        with pytest.raises(SandboxViolation):
            terminate_pid(sandbox, 1)


class TestAppTools:
    def test_discover_and_find(self, tmp_path):
        d = tmp_path / "applications"
        d.mkdir()
        (d / "org.mozilla.firefox.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=Firefox\nExec=firefox %u\n")
        (d / "hidden.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=Hidden\nExec=x\nNoDisplay=true\n")
        apps = discover_apps([d])
        assert "hidden" not in apps
        assert find_app("firefox", apps).name == "Firefox"
        assert find_app("Firefox", apps).app_id == "org.mozilla.firefox"

    def test_unknown_app_not_launched(self, sandbox):
        t = tools_by_name(make_app_tools(sandbox))
        assert "No installed application" in t["launch_application"](name="zzz-not-an-app-zzz")


class TestSecurityTools:
    def test_scan_sections(self, sandbox):
        t = tools_by_name(make_security_tools(sandbox))
        out = t["security_scan"]()
        for section in ("Firewall", "Listening ports", "Suspicious processes", "Agent sandbox"):
            assert section in out


class ScriptedLLM(BaseLLM):
    """Returns queued responses; records every request."""

    def __init__(self, responses):
        super().__init__(model="scripted")
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages):
        self.requests.append(messages)
        return self.responses.pop(0)

    chat_with_tools = lambda self, messages, tools: self.chat(messages)


class TestSmartOS:
    def test_routes_to_specialist_and_uses_tool(self, sandbox):
        llm = ScriptedLLM([
            LLMResponse(content='[{"agent": "FileManager", "task": "write hi to a.txt"}]'),
            LLMResponse(tool_calls=[{"id": "1", "name": "write_file",
                                     "arguments": {"path": "a.txt", "content": "hi"}}]),
            LLMResponse(content="Done, wrote a.txt"),
        ])
        smart = SmartOS(llm=llm, sandbox=sandbox)
        assert smart.run("save hi to a file") == "Done, wrote a.txt"
        assert (sandbox.policy.workspace / "a.txt").read_text() == "hi"

    def test_bad_router_output_falls_back_to_generalist(self, sandbox):
        llm = ScriptedLLM([LLMResponse(content="not json"), LLMResponse(content="hello!")])
        smart = SmartOS(llm=llm, sandbox=sandbox)
        assert smart.run("hi") == "hello!"
        assert smart.generalist.memory.get_messages()[-1]["content"] == "hello!"

    def test_violation_reaches_llm_as_error(self, sandbox):
        sandbox.confirmer = deny_all
        llm = ScriptedLLM([
            LLMResponse(content='[{"agent": "TaskManager", "task": "kill pid 1"}]'),
            LLMResponse(tool_calls=[{"id": "1", "name": "terminate_process",
                                     "arguments": {"pid": 1}}]),
            LLMResponse(content="I can't stop that process."),
        ])
        smart = SmartOS(llm=llm, sandbox=sandbox)
        smart.run("kill init")
        tool_msg = [m for m in llm.requests[-1] if m["role"] == "tool"][-1]
        assert "SandboxViolation" in tool_msg["content"]

    def test_single_mode_skips_router(self, sandbox):
        llm = ScriptedLLM([LLMResponse(content="ok")])
        smart = SmartOS(llm=llm, sandbox=sandbox, mode="single")
        assert smart.run("hi") == "ok" and len(llm.requests) == 1

    def test_history_given_to_router(self, sandbox):
        llm = ScriptedLLM([
            LLMResponse(content='[{"agent": "Generalist", "task": "hi"}]'),
            LLMResponse(content="first answer"),
            LLMResponse(content='[{"agent": "Generalist", "task": "again"}]'),
            LLMResponse(content="second"),
        ])
        smart = SmartOS(llm=llm, sandbox=sandbox)
        smart.run("hello")
        smart.run("and again")
        assert "first answer" in llm.requests[2][0]["content"]
