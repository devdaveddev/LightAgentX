"""Every tool call of a sandboxed agent passes through the sandbox (option 2)."""

import json
from pathlib import Path

import pytest

from lightagentx import Risk, Sandbox, SandboxPolicy, SingleAgent, tool
from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.sandbox import SubprocessBackend, allow_all, deny_all


class CallOnce(BaseLLM):
    """Asks for one tool call, then repeats whatever the tool returned."""

    def __init__(self, name, args):
        super().__init__(model="scripted")
        self.call, self.done = (name, args), False

    def chat(self, messages):
        return LLMResponse(content="ok")

    def chat_with_tools(self, messages, tools):
        if not self.done:
            self.done = True
            return LLMResponse(tool_calls=[{"id": "c1", "name": self.call[0], "arguments": self.call[1]}])
        return LLMResponse(content=[m for m in messages if m["role"] == "tool"][-1]["content"])


@pytest.fixture
def env(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "key.txt").write_text("TOP-SECRET")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("hello")
    asked = []
    sandbox = Sandbox(
        policy=SandboxPolicy(workspace=tmp_path / "ws", read_paths=[tmp_path], deny_paths=[secrets]),
        backend=SubprocessBackend(),
        confirmer=lambda d, r: asked.append((d, r)) or False,
    )
    return {"root": tmp_path, "secrets": secrets, "docs": docs, "sandbox": sandbox, "asked": asked}


def run(tools, name, args, sandbox=None):
    agent = SingleAgent(name="Mine", llm=CallOnce(name, args), tools=tools,
                        sandbox=sandbox, verbose=False)
    return agent.run("go")


calls = []


@tool
def raw_read(path: str) -> str:
    """Read a file (no declarations).

    Args:
        path: File path.
    """
    calls.append(path)
    return open(path).read()


@tool(risk="low", reads=["path"])
def safe_read(path: str) -> str:
    """Read a file (declared).

    Args:
        path: File path.
    """
    calls.append(path)
    return open(path).read()


@tool(risk="medium", writes=["path"])
def save(path: str, text: str) -> str:
    """Write a file.

    Args:
        path: File path.
        text: Content.
    """
    calls.append(path)
    Path(path).write_text(text)
    return "saved"


@tool(risk="low")
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: First.
        b: Second.
    """
    return a + b


@pytest.fixture(autouse=True)
def _reset_calls():
    calls.clear()


class TestDecorator:
    def test_bare_and_configured_forms(self):
        assert raw_read.risk is None and raw_read.reads == ()
        assert safe_read.risk == Risk.LOW and safe_read.reads == ("path",)
        assert save.writes == ("path",)

    def test_rejects_unknown_path_argument(self):
        with pytest.raises(ValueError, match="no such parameter"):
            @tool(reads=["nope"])
            def f(path: str) -> str:
                """F."""
                return path

    def test_rejects_unknown_risk(self):
        with pytest.raises(ValueError):
            @tool(risk="extreme")
            def g(x: str) -> str:
                """G."""
                return x


class TestWithoutSandbox:
    def test_unchanged_behaviour(self, env):
        out = run([raw_read], "raw_read", {"path": str(env["secrets"] / "key.txt")})
        assert out == "TOP-SECRET"  # no sandbox attached -> plain Python, as before


class TestGate:
    def test_the_old_gap_is_closed_undeclared_tool_needs_a_human(self, env):
        out = run([raw_read], "raw_read", {"path": str(env["secrets"] / "key.txt")}, env["sandbox"])
        assert "SandboxViolation" in out and "declined" in out
        assert calls == []                                   # the function never ran
        (desc, risk), = env["asked"]
        assert risk == Risk.HIGH and "risk not declared" in desc

    def test_declared_read_of_denied_path_is_blocked_without_asking(self, env):
        out = run([safe_read], "safe_read", {"path": str(env["secrets"] / "key.txt")}, env["sandbox"])
        assert "Read access" in out and calls == [] and env["asked"] == []

    def test_declared_low_risk_tool_runs_freely(self, env):
        out = run([safe_read], "safe_read", {"path": str(env["docs"] / "notes.txt")}, env["sandbox"])
        assert out == "hello" and env["asked"] == []
        assert env["sandbox"].audit_log[-1].decision == "allowed"

    def test_relative_path_is_resolved_before_the_tool_sees_it(self, env, monkeypatch):
        # cwd contains a different notes.txt; the tool must open the workspace one it was approved for
        monkeypatch.chdir(env["docs"])
        ws = env["sandbox"].policy.workspace
        (ws / "notes.txt").write_text("workspace copy")
        out = run([safe_read], "safe_read", {"path": "notes.txt"}, env["sandbox"])
        assert out == "workspace copy"
        assert calls == [str(ws / "notes.txt")]

    def test_write_outside_allowed_places_is_blocked(self, env):
        out = run([save], "save", {"path": str(env["docs"] / "x.txt"), "text": "x"}, env["sandbox"])
        assert "Write access" in out and not (env["docs"] / "x.txt").exists()

    def test_write_inside_workspace_runs_after_policy_check(self, env):
        out = run([save], "save", {"path": "out.txt", "text": "x"}, env["sandbox"])
        assert out == "saved"
        assert (env["sandbox"].policy.workspace / "out.txt").read_text() == "x"

    def test_human_approval_lets_a_risky_tool_run(self, env):
        env["sandbox"].confirmer = allow_all
        out = run([raw_read], "raw_read", {"path": str(env["docs"] / "notes.txt")}, env["sandbox"])
        assert out == "hello"
        assert env["sandbox"].audit_log[-1].decision == "approved"

    def test_every_call_is_audited_with_arguments(self, env):
        run([add], "add", {"a": 2, "b": 3}, env["sandbox"])
        entry = env["sandbox"].audit_log[-1]
        assert entry.action == "tool:add" and json.loads(entry.target) == {"a": 2, "b": 3}

    def test_policy_can_lower_the_default_for_undeclared_tools(self, env):
        env["sandbox"].policy.undeclared_tool_risk = Risk.LOW
        out = run([raw_read], "raw_read", {"path": str(env["docs"] / "notes.txt")}, env["sandbox"])
        assert out == "hello" and env["asked"] == []


class TestBuiltInToolsNotDoubleChecked:
    def test_smartos_tool_asks_once(self, env):
        pytest.importorskip("psutil")
        from lightagentx.smartos import make_file_tools
        sb = env["sandbox"]
        target = sb.policy.workspace / "old.txt"
        target.write_text("x")
        tools = make_file_tools(sb)
        assert all(t.guarded for t in tools)
        run(tools, "move_to_trash", {"path": str(target)}, sb)
        assert len(env["asked"]) == 1          # the tool's own confirmation, not a second gate
        assert target.exists()                 # declined -> nothing deleted


class TestRegistrySessions:
    def test_attach_with_sandbox_gates_tools(self, env):
        from lightagentx.state import AgentRegistry
        reg = AgentRegistry(env["root"] / "agents")
        aid = reg.create("Bot", owner="me")
        with reg.attach(aid, "me", llm=CallOnce("raw_read", {"path": str(env["secrets"] / "key.txt")}),
                        tools=[raw_read], sandbox=env["sandbox"]) as s:
            out = s.run("go")
        assert "SandboxViolation" in out and calls == []

    def test_state_tools_do_not_prompt(self, env):
        from lightagentx.state import AgentRegistry
        reg = AgentRegistry(env["root"] / "agents")
        aid = reg.create("Bot", owner="me")
        llm = CallOnce("set_state", {"key": "customer", "value_json": "\"ACME\""})
        with reg.attach(aid, "me", llm=llm, sandbox=env["sandbox"], state_tools=True) as s:
            s.run("remember")
        assert env["asked"] == [] and reg.read(aid, "me")["state"]["customer"] == "ACME"


def test_policy_round_trip_keeps_undeclared_tool_risk(tmp_path):
    p = SandboxPolicy(workspace=tmp_path, undeclared_tool_risk=Risk.MEDIUM)
    assert SandboxPolicy.from_dict(p.to_dict()).undeclared_tool_risk == Risk.MEDIUM
