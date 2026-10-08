"""lightx doctor: checks, verified fixes, migrations. Uses a scripted LLM (no network)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightagentx.doctor import migrate_snapshot_data, run_checks, run_fixes, snapshot_layout
from lightagentx.doctor.checks import default_test_command
from lightagentx.doctor.project import Workspace
from lightagentx.llm.base import BaseLLM, LLMResponse

ROOT = Path(__file__).resolve().parents[1]

FAKELIB = '''
import warnings
def compute(x):
    warnings.warn("compute() is deprecated; use compute_v2()", DeprecationWarning, stacklevel=2)
    return compute_v2(x)
def compute_v2(x):
    return x * 2
'''
APP = '''import fakelib
from fakelib import load_data
from fakelib.legacy import helper


def report(name):
    data = load_data(name)
    return f"{helper(data['name'])}: {fakelib.compute(data['rows'])} rows"
'''
TEST = '''from app import report


def test_report():
    assert report("sales") == "SALES: 6 rows"
'''
LEGACY_SNAPSHOT = {
    "format_version": 1, "timestamp": 1,
    "agent": {"name": "OldBot", "description": "", "system_prompt": "Be brief.", "max_iterations": 10,
              "verbose": False},
    "llm_config": {"model": "x", "temperature": 0.7, "max_tokens": 1024}, "tool_manifest": [],
    "memory": {"type": "SummaryMemory", "messages": [
        {"role": "system", "content": "Be brief.\n\nCONVERSATION SUMMARY SO FAR:\nACME wants a refund."},
        {"role": "user", "content": "and the invoice?"}, {"role": "assistant", "content": "Sent."}]},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    libs = tmp_path / "libs"
    (libs / "fakelib").mkdir(parents=True)
    (libs / "fakelib" / "__init__.py").write_text(FAKELIB)
    (libs / "fakelib" / "io.py").write_text("def load_data(name):\n    return {'name': name, 'rows': 3}\n")
    (libs / "fakelib_community").mkdir()
    (libs / "fakelib_community" / "__init__.py").write_text("")
    (libs / "fakelib_community" / "legacy.py").write_text("def helper(s):\n    return s.upper()\n")
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "app.py").write_text(APP)
    (proj / "tests" / "test_app.py").write_text(TEST)
    (proj / "requirements.txt").write_text("packaging>=999\nsurely-not-installed-pkg>=1\n")
    (proj / "old.agent.json").write_text(json.dumps(LEGACY_SNAPSHOT))
    monkeypatch.setenv("PYTHONPATH", f"{libs}")
    monkeypatch.syspath_prepend(str(libs))
    ws = Workspace(proj, backend="auto", timeout_s=120, base_dir=tmp_path / "work")
    yield {"proj": proj, "ws": ws, "tmp": tmp_path}
    ws.cleanup()


def check(env):
    return run_checks(env["proj"], ws=env["ws"], test_command=default_test_command(env["proj"]),
                      agent_paths=[])


class ScriptedFixer(BaseLLM):
    """Returns the right patch for each known problem (or a configured wrong one)."""

    def __init__(self, wrong_first=False, edit_tests=False):
        super().__init__(model="scripted")
        self.calls, self.wrong_first, self.edit_tests = 0, wrong_first, edit_tests

    def chat(self, messages):
        self.calls += 1
        prompt = messages[-1]["content"]
        if self.edit_tests:
            edits = [{"file": "tests/test_app.py", "search": "SALES", "replace": "X"}]
        elif self.wrong_first and self.calls == 1:
            edits = [{"file": "app.py", "search": "from fakelib import load_data",
                      "replace": "from fakelib.nowhere import load_data"}]
        elif "'load_data' no longer exists" in prompt:
            edits = [{"file": "app.py", "search": "from fakelib import load_data",
                      "replace": "from fakelib.io import load_data"}]
        elif "fakelib.legacy" in prompt:
            edits = [{"file": "app.py", "search": "from fakelib.legacy import helper",
                      "replace": "from fakelib_community.legacy import helper"}]
        elif "compute()" in prompt:
            edits = [{"file": "app.py", "search": "fakelib.compute(", "replace": "fakelib.compute_v2("}]
        else:
            edits = []
        return LLMResponse(content=json.dumps({"explanation": "scripted", "edits": edits}))

    chat_with_tools = lambda self, m, t: self.chat(m)


class TestChecks:
    def test_finds_every_kind_of_problem(self, env):
        findings, ctx = check(env)
        titles = {f.title for f in findings}
        assert "'load_data' no longer exists in fakelib" in titles
        assert "Can't import fakelib.legacy" in titles
        assert any("packaging" in t and "999" in t for t in titles)
        assert "surely-not-installed-pkg is not installed" in titles
        assert "Snapshot in the old layout" in titles
        assert ctx.baseline.ran and ctx.baseline.failed      # tests can't even be collected yet

    def test_points_to_where_things_moved(self, env):
        findings, _ = check(env)
        by_title = {f.title: f for f in findings}
        assert by_title["'load_data' no longer exists in fakelib"].data["candidates"] == ["fakelib.io"]
        assert "fakelib_community.legacy" in by_title["Can't import fakelib.legacy"].data["candidates"]
        assert by_title["Can't import fakelib.legacy"].where == "app.py:3"

    def test_check_only_never_changes_the_project(self, env):
        before = {p: p.read_bytes() for p in env["proj"].rglob("*") if p.is_file()}
        check(env)
        assert {p: p.read_bytes() for p in env["proj"].rglob("*") if p.is_file()} == before

    def test_project_code_runs_only_in_the_copy(self, env):
        (env["proj"] / "app.py").write_text(APP + "\nopen('RAN_IN_PROJECT', 'w').write('x')\n")
        check(env)
        assert not (env["proj"] / "RAN_IN_PROJECT").exists()


class TestFixes:
    def run(self, env, llm, approve=lambda q: True):
        findings, ctx = check(env)
        code = [f for f in findings if f.fixable == "code"]
        return run_fixes(code, ctx, llm=llm, test_command=default_test_command(env["proj"]),
                         approve=approve, say=lambda s: None, backup_root=env["tmp"] / "backups")

    def test_verified_patches_are_applied_and_backed_up(self, env):
        log = self.run(env, ScriptedFixer())
        assert [e["outcome"] for e in log] == ["applied", "applied"]
        app = (env["proj"] / "app.py").read_text()
        assert "from fakelib.io import load_data" in app
        assert "from fakelib_community.legacy import helper" in app
        backups = list((env["tmp"] / "backups").rglob("app.py"))
        assert backups and backups[0].read_text() == APP
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=env["proj"], capture_output=True, text=True)
        assert "1 passed" in r.stdout

    def test_second_round_fixes_the_revealed_deprecation(self, env):
        self.run(env, ScriptedFixer())
        findings, ctx = check(env)
        dep = [f for f in findings if f.check == "deprecation"]
        assert dep and dep[0].where == "app.py:8"
        log = run_fixes(dep, ctx, llm=ScriptedFixer(), test_command=default_test_command(env["proj"]),
                        approve=lambda q: True, say=lambda s: None, backup_root=env["tmp"] / "b2")
        assert log[0]["outcome"] == "applied"
        assert "compute_v2(" in (env["proj"] / "app.py").read_text()

    def test_declined_patch_is_not_applied(self, env):
        log = self.run(env, ScriptedFixer(), approve=lambda q: False)
        assert {e["outcome"] for e in log} == {"skipped by you"}
        assert (env["proj"] / "app.py").read_text() == APP

    def test_wrong_patch_fails_verification_then_retries(self, env):
        llm = ScriptedFixer(wrong_first=True)
        log = self.run(env, llm)
        assert log[0]["outcome"] == "applied" and llm.calls >= 3   # 1 wrong + 1 right + 1 for the next

    def test_patches_to_tests_are_refused(self, env):
        log = self.run(env, ScriptedFixer(edit_tests=True))
        assert all(e["outcome"] == "not fixed" for e in log)
        assert "SALES" in (env["proj"] / "tests" / "test_app.py").read_text()

    def test_without_llm_nothing_changes(self, env):
        log = self.run(env, None)
        assert all("no LLM" in e["detail"] for e in log)
        assert (env["proj"] / "app.py").read_text() == APP


class TestMigrations:
    def test_legacy_snapshot_detected_and_migrated(self):
        assert snapshot_layout(LEGACY_SNAPSHOT) == "legacy"
        new = migrate_snapshot_data(LEGACY_SNAPSHOT)
        assert snapshot_layout(new) == "current"
        assert new["memory"]["summary"] == "ACME wants a refund."
        assert [m["role"] for m in new["memory"]["messages"]] == ["user", "assistant"]

    def test_migration_applied_with_approval_and_backup(self, env):
        findings, ctx = check(env)
        mig = [f for f in findings if f.fixable == "migration"]
        log = run_fixes(mig, ctx, llm=None, test_command=None, approve=lambda q: True,
                        say=lambda s: None, backup_root=env["tmp"] / "backups")
        assert log[0]["outcome"] == "applied"
        from lightagentx import SingleAgent
        agent = SingleAgent.from_snapshot(env["proj"] / "old.agent.json", llm=ScriptedFixer())
        assert agent.memory.summary == "ACME wants a refund."
        assert json.loads(next((env["tmp"] / "backups").rglob("old.agent.json")).read_text()) == LEGACY_SNAPSHOT

    def test_unversioned_registry_detected_and_marked(self, env):
        from lightagentx.state import AgentRegistry
        reg_dir = env["tmp"] / "oldreg"
        AgentRegistry(reg_dir).create("Legacy", owner="me")
        (reg_dir / "FORMAT").unlink()
        findings, ctx = run_checks(env["proj"], ws=env["ws"], test_command=None,
                                   agent_paths=[reg_dir], run_tests_too=False)
        mig = [f for f in findings if f.fixable == "migration" and f.data["kind"] == "registry"]
        assert mig
        run_fixes(mig, ctx, llm=None, test_command=None, approve=lambda q: True, say=lambda s: None,
                  backup_root=env["tmp"] / "backups")
        assert AgentRegistry(reg_dir).store.format_version() == 1


class TestCli:
    def test_json_report_and_exit_code(self, env):
        r = subprocess.run([sys.executable, "-m", "lightagentx", "doctor", str(env["proj"]), "--json",
                            "--no-tests"], capture_output=True, text=True, cwd=ROOT,
                           env={**__import__("os").environ, "PYTHONPATH": f"{ROOT}:{env['tmp'] / 'libs'}"})
        assert r.returncode == 1                     # errors present
        data = json.loads(r.stdout)
        assert any(f["check"] == "import" and f["file"] == "app.py" for f in data)
