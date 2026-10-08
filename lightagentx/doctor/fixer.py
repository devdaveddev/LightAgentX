"""`lightx doctor --fix`: propose, verify in the sandbox, show, apply only with approval."""

from __future__ import annotations

import difflib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..llm.base import BaseLLM
from .checks import CheckContext, TestRun, run_tests
from .migrate import MigrationPlan, apply_migration, plan_registry, plan_snapshot
from .project import Finding, Workspace, collect_imports, line_excerpt

Approver = Callable[[str], bool]

_SYSTEM = """You fix Python code that broke because an installed library changed its API
(a version upgrade or downgrade). Make the smallest change that fixes the problem.

Reply with ONLY a JSON object, no markdown:
{"explanation": "<one or two sentences>",
 "edits": [{"file": "<project-relative path>", "search": "<exact existing text>", "replace": "<new text>"}]}

Rules:
- "search" must be copied exactly from the file and match exactly once; include enough lines.
- Only change project files shown to you. Never edit tests.
- Prefer the library's new location or API (listed under "Where it lives now") over workarounds.
- If you cannot fix it safely, return {"explanation": "<why>", "edits": []}."""


@dataclass
class Proposal:
    finding: Finding
    explanation: str = ""
    edits: list[dict] = field(default_factory=list)
    diff: str = ""
    verified: bool = False
    verification: str = ""
    attempts: int = 0
    new_files: dict[str, str] = field(default_factory=dict)   # relative path -> patched content


# ── LLM patch proposals ───────────────────────────────────────────────────

def _prompt(f: Finding, project: Path, previous: str | None) -> str:
    d = f.data
    parts = [f"Problem: {f.title}", f"Details: {f.detail}"]
    if d.get("package_version"):
        parts.append(f"Installed version of the library: {d['package_version']}")
    if d.get("candidates"):
        parts.append("Where it lives now (found in the installed library):\n- "
                     + "\n- ".join(d["candidates"]))
    if f.file:
        parts.append(f"File {f.file} (line numbers for reference only; do not include them):\n"
                     + line_excerpt(project / f.file, f.line))
    if previous:
        parts.append(f"Your previous attempt did not pass verification:\n{previous}\nTry a different fix.")
    return "\n\n".join(parts)


def _parse(reply: str) -> tuple[str, list[dict]]:
    text = reply.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    edits = [e for e in data.get("edits", []) if {"file", "search", "replace"} <= set(e)]
    return data.get("explanation", ""), edits


def _apply_edits(base: Path, edits: list[dict]) -> dict[str, str]:
    """Apply search/replace edits to files under `base`. Returns {relpath: new content}."""
    changed: dict[str, str] = {}
    for e in edits:
        rel = Path(e["file"])
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] in ("tests", "test"):
            raise ValueError(f"edit targets a disallowed path: {e['file']}")
        path = base / rel
        text = changed.get(str(rel)) or path.read_text(encoding="utf-8")
        count = text.count(e["search"])
        if count != 1:
            raise ValueError(f"search text matches {count} times in {rel} (must be exactly once)")
        changed[str(rel)] = text.replace(e["search"], e["replace"], 1)
    for rel, text in changed.items():
        (base / rel).write_text(text, encoding="utf-8")
    return changed


def _breakage_keys(results: dict) -> set[tuple[str, str]]:
    keys = set()
    for r in results.values():
        if r["error"]:
            keys.add((r["module"], "*"))
        keys |= {(r["module"], n) for n in r["missing"]}
    return keys


def _finding_key(f: Finding) -> tuple[str, str]:
    return (f.data.get("module", ""), f.data.get("name", "*"))


def _still_broken(f: Finding, ws: Workspace, copy: Path, tests: TestRun,
                  known: set[tuple[str, str]] | None = None) -> str | None:
    """
    Re-check the patched copy: THIS problem must be gone, and the patch must not
    break any import that worked before. Other pre-existing problems in the same
    file are ignored here; they get their own fix.
    """
    if f.check == "import" or (f.check == "deprecation" and "module" in f.data):
        uses = [u for u in collect_imports(copy) if u.file == f.file]
        results = ws.probe_imports(copy, uses)
        broken = _breakage_keys(results)
        if f.check == "import" and _finding_key(f) in broken:
            module, name = _finding_key(f)
            return (f"import still fails: {module}" if name == "*"
                    else f"'{name}' still missing in {module}")
        newly = broken - (known or set()) - {_finding_key(f)}
        if newly:
            return "patch breaks other imports: " + ", ".join(
                m if n == "*" else f"{m}.{n}" for m, n in sorted(newly))
        if f.check == "deprecation":
            for r in results.values():
                if f.data.get("warning") in r["warnings"]:
                    return f"warning still raised: {f.data['warning']}"
        return None
    if f.check == "deprecation":
        for w in tests.warnings:
            if w["file"] == f.file and f"{w['category']}: {w['message']}" == f.data.get("warning"):
                return f"warning still raised at {w['file']}:{w['line']}"
        return None
    return None


def propose_code_fix(f: Finding, ctx: CheckContext, llm: BaseLLM, test_command: list[str] | None,
                     max_attempts: int = 2) -> Proposal:
    ws = ctx.workspace
    p = Proposal(f)
    previous = None
    for attempt in range(1, max_attempts + 1):
        p.attempts = attempt
        try:
            reply = llm.chat([{"role": "system", "content": _SYSTEM},
                              {"role": "user", "content": _prompt(f, ctx.project, previous)}]).content
            p.explanation, p.edits = _parse(reply)
        except Exception as e:  # noqa: BLE001 - bad JSON or provider error -> retry
            previous = f"Your reply could not be used: {type(e).__name__}: {e}"
            continue
        if not p.edits:
            p.verification = f"LLM declined: {p.explanation}"
            return p

        copy = ws.fresh_copy(f"attempt-{attempt}")
        try:
            p.new_files = _apply_edits(copy, p.edits)
            for rel in p.new_files:
                py_compile.compile(str(copy / rel), doraise=True)
        except (ValueError, OSError, py_compile.PyCompileError) as e:
            previous = f"Patch could not be applied: {e}"
            continue

        tests = run_tests(ws, copy, test_command)
        problem = _still_broken(f, ws, copy, tests, known=getattr(ctx, "known_breakage", set()))
        new_failures = sorted(tests.failed - ctx.baseline.failed) if tests.ran else []
        if problem:
            previous = problem
        elif new_failures:
            previous = "These tests fail after your patch but passed before:\n" + "\n".join(new_failures[:10]) \
                + "\n" + tests.output[-3000:]
        else:
            p.verified = True
            p.verification = ("problem gone; " + (
                f"tests: no new failures ({len(tests.failed)} failing before and after)"
                if tests.ran and tests.failed else "tests pass" if tests.ran
                else "no tests to run (verified by re-checking only)"))
            p.diff = _diff(ctx.project, p.new_files)
            return p
        p.verification = f"attempt {attempt} failed verification: {previous.splitlines()[0]}"
    p.diff = _diff(ctx.project, p.new_files) if p.new_files else ""
    return p


def _diff(project: Path, new_files: dict[str, str]) -> str:
    out = []
    for rel, new in new_files.items():
        old = (project / rel).read_text(encoding="utf-8")
        out += difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
    return "".join(out)


# ── dependency proposals: verified in an overlay before touching the environment ──

def propose_dependency_fix(f: Finding, ctx: CheckContext, test_command: list[str] | None) -> Proposal:
    ws = ctx.workspace
    spec = f.data["install"]
    p = Proposal(f, explanation=f"pip install \"{spec}\"")
    overlay = ws.root / "overlay"
    shutil.rmtree(overlay, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                        "--target", str(overlay), spec], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        p.verification = "could not install it for verification (network?): " + (r.stderr.strip().splitlines() or ["?"])[-1]
        return p
    copy = ws.fresh_copy("dependency")
    original_env = ws.env_for
    ws.env_for = lambda c: {**original_env(c), "PYTHONPATH": f"{overlay}{os.pathsep}{original_env(c)['PYTHONPATH']}"}
    try:
        tests = run_tests(ws, copy, test_command)
        results = ws.probe_imports(copy, collect_imports(copy))
    finally:
        ws.env_for = original_env
    broken = [r for r in results.values() if r["error"] or r["missing"]]
    new_failures = sorted(tests.failed - ctx.baseline.failed) if tests.ran else []
    if new_failures or len(broken) > _broken_count(ctx):
        p.verification = ("with this version, " + (f"{len(new_failures)} test(s) newly fail"
                          if new_failures else f"{len(broken)} import(s) break"))
        return p
    p.verified = True
    p.verification = "installed in a throwaway overlay: imports work and " + (
        "no new test failures" if tests.ran else "no tests to run")
    p.diff = f"$ {sys.executable} -m pip install \"{spec}\"\n"
    return p


def _broken_count(ctx: CheckContext) -> int:
    return getattr(ctx, "_broken_imports", 0)


def apply_dependency(p: Proposal) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", "install", p.finding.data["install"]],
                          capture_output=True, text=True, timeout=900)


# ── orchestration ────────────────────────────────────────────────────────

def backup_dir(root: Path | None = None) -> Path:
    base = root or Path.home() / ".lightx" / "doctor" / "backups"
    d = base / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def apply_code(p: Proposal, project: Path, backups: Path) -> None:
    for rel, text in p.new_files.items():
        target = project / rel
        dest = backups / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():  # a later patch to the same file must not overwrite the original
            dest.write_bytes(target.read_bytes())
        target.write_text(text, encoding="utf-8")


def plan_migration(f: Finding) -> MigrationPlan:
    path = Path(f.data["path"])
    return plan_snapshot(path) if f.data["kind"] == "snapshot" else plan_registry(path)


def run_fixes(findings: list[Finding], ctx: CheckContext, *, llm: BaseLLM | None,
              test_command: list[str] | None, approve: Approver, say: Callable[[str], None],
              max_attempts: int = 2, backup_root: Path | None = None) -> list[dict]:
    """Fix what can be fixed. Every change needs `approve(...)` -> True. Returns an outcome log."""
    log: list[dict] = []
    backups: Path | None = None
    ctx._broken_imports = sum(1 for f in findings if f.check == "import")  # type: ignore[attr-defined]
    # Import problems that already exist; a patch may leave these for their own fix.
    ctx.known_breakage = {_finding_key(f) for f in findings if f.check == "import"}  # type: ignore[attr-defined]

    def record(f: Finding, outcome: str, detail: str = "") -> None:
        log.append({"finding": f.title, "where": f.where, "outcome": outcome, "detail": detail})

    for f in [f for f in findings if f.fixable == "migration"]:
        plan = plan_migration(f)
        say(f"\n▸ {f.title} — {f.where}\n  {plan.description}\n  verification: {plan.verification}")
        if not plan.verified:
            record(f, "not fixed", plan.verification)
            continue
        if approve(f"Apply migration to {f.where}?"):
            backups = backups or backup_dir(backup_root)
            apply_migration(plan, backups)
            record(f, "applied", plan.verification)
        else:
            record(f, "skipped by you")

    for f in [f for f in findings if f.fixable == "dependency"]:
        say(f"\n▸ {f.title}\n  proposed: pip install \"{f.data['install']}\"  (verifying in a throwaway overlay…)")
        p = propose_dependency_fix(f, ctx, test_command)
        say(f"  verification: {p.verification}")
        if not p.verified:
            record(f, "not fixed", p.verification)
            continue
        if approve(f"Run: pip install \"{f.data['install']}\" in your environment?"):
            r = apply_dependency(p)
            record(f, "applied" if r.returncode == 0 else "failed", (r.stderr or r.stdout)[-300:])
        else:
            record(f, "skipped by you")

    code = [f for f in findings if f.fixable == "code"]
    if code and llm is None:
        for f in code:
            record(f, "not fixed", "no LLM configured (set an API key or pass --base-url)")
        say("\n(Code problems need an LLM: set an API key or pass --base-url.)")
        return log

    for f in code:
        if f.check == "import" and _resolved_now(f, ctx):
            say(f"\n▸ {f.title} — {f.where}\n  already fixed by an earlier patch")
            record(f, "already fixed")
            continue
        say(f"\n▸ {f.title} — {f.where}\n  asking the LLM for a patch and verifying it in the sandbox…")
        p = propose_code_fix(f, ctx, llm, test_command, max_attempts=max_attempts)
        say(f"  {p.explanation}\n  verification ({p.attempts} attempt(s)): {p.verification}")
        if p.diff:
            say(_indent(p.diff))
        if not p.verified:
            record(f, "not fixed", p.verification)
            continue
        if approve(f"Apply this patch to {', '.join(p.new_files)}?"):
            backups = backups or backup_dir(backup_root)
            apply_code(p, ctx.project, backups)
            record(f, "applied", p.verification)
        else:
            record(f, "skipped by you")

    if backups and any(e["outcome"] == "applied" for e in log):
        say(f"\nOriginals backed up in {backups}")
    return log


def _resolved_now(f: Finding, ctx: CheckContext) -> bool:
    """Has an earlier applied patch already fixed this finding?"""
    copy = ctx.workspace.fresh_copy("recheck")
    return _still_broken(f, ctx.workspace, copy, TestRun(),
                         known=getattr(ctx, "known_breakage", set())) is None


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.rstrip().splitlines())
