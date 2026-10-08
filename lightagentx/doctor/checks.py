"""The doctor's checks. Read-only: they inspect the project and a sandboxed copy of it."""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .project import (Finding, ImportUse, Workspace, collect_imports, find_definitions,
                      find_module_candidates, iter_files)

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version as PkgVersion
except ImportError:  # pragma: no cover - packaging ships with pip almost everywhere
    Requirement = None


@dataclass
class TestRun:
    ran: bool = False
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    failed: set[str] = field(default_factory=set)        # node ids / collection errors
    warnings: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""


@dataclass
class CheckContext:
    project: Path
    workspace: Workspace | None = None
    baseline: TestRun = field(default_factory=TestRun)
    import_uses: list[ImportUse] = field(default_factory=list)


# ── 1. dependencies ───────────────────────────────────────────────────────

def _read_requirements(project: Path) -> list[tuple[str, str]]:
    """(requirement string, source file)."""
    reqs: list[tuple[str, str]] = []
    pyproject = project / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:  # Python 3.10
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                tomllib = None
        if tomllib is not None:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            reqs += [(r, "pyproject.toml") for r in data.get("project", {}).get("dependencies", [])]
    for f in sorted(project.glob("requirements*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.split(" #")[0].strip()
            if line and not line.startswith(("#", "-", "git+", "http://", "https://")):
                reqs.append((line, f.name))
    return reqs


def _installed_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check_dependencies(ctx: CheckContext) -> list[Finding]:
    findings: list[Finding] = []
    if Requirement is None:
        return [Finding("dependency", "info", "Version checks skipped",
                        "Install 'packaging' to compare installed and required versions.")]
    sources = _read_requirements(ctx.project)
    try:
        own = md.requires("lightagentx") or []
        sources += [(r, "lightagentx itself") for r in own]
    except md.PackageNotFoundError:
        pass

    seen = set()
    for text, source in sources:
        try:
            req = Requirement(text)
        except InvalidRequirement:
            findings.append(Finding("dependency", "warning", f"Unparseable requirement '{text}'",
                                    f"in {source}", file=source))
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        key = (req.name.lower(), str(req.specifier))
        if key in seen:
            continue
        seen.add(key)
        installed = _installed_version(req.name)
        if installed is None:
            findings.append(Finding(
                "dependency", "error", f"{req.name} is not installed",
                f"required by {source}: {text}", file=source if source != "lightagentx itself" else None,
                fixable="dependency", data={"install": str(req).split(";")[0].strip(), "package": req.name}))
            continue
        try:
            ok = not req.specifier or req.specifier.contains(PkgVersion(installed), prereleases=True)
        except InvalidVersion:
            ok = True
        if not ok:
            findings.append(Finding(
                "dependency", "error", f"{req.name} {installed} doesn't satisfy '{req.specifier}'",
                f"required by {source}", file=source if source != "lightagentx itself" else None,
                fixable="dependency",
                data={"install": f"{req.name}{req.specifier}", "package": req.name, "installed": installed}))
    findings += _pip_check()
    return findings


_PIP_HAS = re.compile(r"^(\S+) (\S+) has requirement (.+), but you have (\S+) (\S+)\.$")
_PIP_MISSING = re.compile(r"^(\S+) (\S+) requires (\S+), which is not installed\.$")


def _pip_check() -> list[Finding]:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        if m := _PIP_HAS.match(line.strip()):
            pkg, ver, need, dep, have = m.groups()
            out.append(Finding("dependency", "warning", f"Conflict: {pkg} {ver} needs {need}",
                               f"but {dep} {have} is installed (pip check)", fixable="dependency",
                               data={"install": need, "package": dep, "installed": have}))
        elif m := _PIP_MISSING.match(line.strip()):
            pkg, ver, dep = m.groups()
            out.append(Finding("dependency", "warning", f"Conflict: {pkg} {ver} needs {dep}",
                               "which is not installed (pip check)", fixable="dependency",
                               data={"install": dep, "package": dep}))
    return out


# ── 2. imports ────────────────────────────────────────────────────────────

def check_imports(ctx: CheckContext, copy: Path) -> list[Finding]:
    uses = collect_imports(ctx.project)
    ctx.import_uses = uses
    unique: dict[tuple[str, tuple[str, ...]], ImportUse] = {}
    for u in uses:
        unique.setdefault((u.module, tuple(u.names)), u)
    results = ctx.workspace.probe_imports(copy, list(unique.values()))

    findings: list[Finding] = []
    for key, res in results.items():
        locs = [u for u in uses if (u.module, tuple(u.names)) == key]
        first = locs[0]
        others = [f"{u.file}:{u.line}" for u in locs[1:]]
        top = res["module"].split(".")[0]
        version = _installed_version(top) or _installed_version(top.replace("_", "-"))
        if res["error"]:
            missing_module = res["error"].startswith("ModuleNotFoundError")
            candidates = find_module_candidates(res["module"]) if missing_module else []
            findings.append(Finding(
                "import", "error", f"Can't import {res['module']}", res["error"]
                + (f"\nPossibly moved to: {', '.join(candidates)}" if candidates else "")
                + (f"\nAlso used at: {', '.join(others)}" if others else ""),
                file=first.file, line=first.line, fixable="code",
                data={"module": res["module"], "names": list(key[1]), "error": res["error"],
                      "candidates": candidates, "package_version": version, "kind": "module"}))
        for name in res["missing"]:
            candidates = find_definitions(top, name)
            findings.append(Finding(
                "import", "error", f"'{name}' no longer exists in {res['module']}",
                (f"{top} {version} is installed" if version else f"{top} is installed (version unknown)")
                + (f"; '{name}' is now defined in: {', '.join(candidates)}" if candidates else "")
                + (f"\nAlso used at: {', '.join(others)}" if others else ""),
                file=first.file, line=first.line, fixable="code",
                data={"module": res["module"], "name": name, "candidates": candidates,
                      "package_version": version, "kind": "name"}))
        for w in res["warnings"]:
            findings.append(Finding(
                "deprecation", "warning", f"Importing {res['module']} is deprecated", w,
                file=first.file, line=first.line, fixable="code",
                data={"module": res["module"], "warning": w, "package_version": version}))
    return findings


# ── 3. tests & deprecations (run on the sandboxed copy) ───────────────────

_WARN_LINE = re.compile(r"^\s*(?P<file>[^\s:]+\.py):(?P<line>\d+): (?P<cat>\w*Warning): (?P<msg>.*)$")
_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)")


def default_test_command(project: Path) -> list[str] | None:
    has_tests = (project / "tests").is_dir() or (project / "test").is_dir() or \
        any(project.glob("test_*.py"))
    if not has_tests:
        return None
    try:
        import pytest  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
            "-W", "always::DeprecationWarning", "-W", "always::PendingDeprecationWarning",
            "-W", "always::FutureWarning"]


def run_tests(ws: Workspace, copy: Path, command: list[str] | None) -> TestRun:
    if not command:
        return TestRun()
    r = ws.run(copy, command)
    out = r.stdout + "\n" + r.stderr
    run = TestRun(ran=True, command=command, exit_code=r.exit_code, output=out[-20000:])
    for line in out.splitlines():
        if m := _FAILED.match(line.strip()):
            run.failed.add(m.group(1))
        if m := _WARN_LINE.match(line):
            f = m.group("file")
            path = (copy / f).resolve()
            try:
                rel = str(path.relative_to(copy.resolve()))
            except ValueError:
                continue  # warning raised inside a library, not the project
            if not (ws.project / rel).exists() or rel.startswith(("tests/", "test/")):
                continue
            run.warnings.append({"file": rel, "line": int(m.group("line")),
                                 "category": m.group("cat"), "message": m.group("msg").strip()})
    if r.timed_out:
        run.failed.add("<timeout>")
    return run


def check_tests(ctx: CheckContext, copy: Path, command: list[str] | None) -> list[Finding]:
    ctx.baseline = run_tests(ctx.workspace, copy, command)
    findings: list[Finding] = []
    if not ctx.baseline.ran:
        return [Finding("tests", "info", "No tests run",
                        "No tests found (or pytest not installed); fixes are verified by re-checking only.")]
    seen = set()
    for w in ctx.baseline.warnings:
        key = (w["file"], w["line"], w["message"])
        if key in seen:
            continue
        seen.add(key)
        findings.append(Finding("deprecation", "warning", f"{w['category']} in your code",
                                w["message"], file=w["file"], line=w["line"], fixable="code",
                                data={"warning": f"{w['category']}: {w['message']}"}))
    if ctx.baseline.failed:
        findings.append(Finding("tests", "info", f"{len(ctx.baseline.failed)} test(s) already failing",
                                "\n".join(sorted(ctx.baseline.failed)[:20]),
                                data={"failed": sorted(ctx.baseline.failed)}))
    return findings


# ── 4. saved agents ───────────────────────────────────────────────────────

def snapshot_layout(data: dict[str, Any]) -> str:
    """'current', 'legacy' (needs migration) or 'unsupported'."""
    if data.get("format_version") != 1:
        return "unsupported"
    mem = data.get("memory")
    if not mem:
        return "current"
    if "max_messages" not in mem or any(m.get("role") == "system" for m in mem.get("messages", [])):
        return "legacy"
    return "current"


def check_saved_agents(ctx: CheckContext, extra_paths: list[Path]) -> list[Finding]:
    from ..state.archive import ARCHIVE_FORMAT, ARCHIVE_VERSION
    from ..state.registry import AgentRegistry
    from ..state.store import IntegrityError

    findings: list[Finding] = []
    roots = [ctx.project, *[p for p in extra_paths if p.is_dir()]]
    files = [p for p in extra_paths if p.is_file()]
    for root in roots:
        files += iter_files(root, "*.agent.json") + iter_files(root, "*.lxagent")
    for f in sorted(set(files)):
        shown = _show(f, ctx.project)
        if f.name.endswith(".agent.json"):
            try:
                layout = snapshot_layout(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                findings.append(Finding("saved-agent", "error", "Unreadable snapshot", str(e), file=shown))
                continue
            if layout == "legacy":
                findings.append(Finding(
                    "saved-agent", "warning", "Snapshot in the old layout",
                    "Restoring it loses its memory type and window size (and a SummaryMemory "
                    "summary). Migrating rewrites it in the current layout.",
                    file=shown, fixable="migration", data={"kind": "snapshot", "path": str(f)}))
            elif layout == "unsupported":
                findings.append(Finding("saved-agent", "error", "Snapshot format not supported",
                                        "Unknown format_version.", file=shown))
        else:
            try:
                with tarfile.open(f, "r:gz") as tar:
                    manifest = json.loads(tar.extractfile("manifest.json").read())
            except (tarfile.TarError, KeyError, OSError, ValueError, AttributeError) as e:
                findings.append(Finding("saved-agent", "error", "Unreadable agent archive", str(e), file=shown))
                continue
            if manifest.get("format") != ARCHIVE_FORMAT or manifest.get("format_version") != ARCHIVE_VERSION:
                findings.append(Finding("saved-agent", "error", "Agent archive format not supported",
                                        f"format_version={manifest.get('format_version')}", file=shown))

    registries = [p for p in roots[1:] if (p / "agents").is_dir()]
    default = Path.home() / ".lightx" / "agents"
    if default.is_dir():
        registries.append(default)
    for root in dict.fromkeys(registries):
        agents_dir = root / "agents"
        ids = [p.name for p in agents_dir.iterdir() if (p / "meta.json").exists()] if agents_dir.is_dir() else []
        if not ids:
            continue
        reg = AgentRegistry(root)
        for aid in ids:
            try:
                reg.verify(aid)
            except IntegrityError as e:
                findings.append(Finding("saved-agent", "error", f"Registry agent {aid} is corrupted",
                                        str(e), file=str(root)))
        if reg.store.format_version() is None:
            findings.append(Finding(
                "saved-agent", "warning", "Registry has no format marker",
                f"Created before registries recorded their format ({len(ids)} agent(s)). Migrating "
                "verifies every agent and records the format so future versions can upgrade it.",
                file=str(root), fixable="migration", data={"kind": "registry", "path": str(root)}))
    return findings


def _show(path: Path, project: Path) -> str:
    try:
        return str(path.relative_to(project))
    except ValueError:
        return str(path)


# ── run everything ────────────────────────────────────────────────────────

def run_checks(project: Path, *, ws: Workspace, test_command: list[str] | None,
               agent_paths: list[Path], run_tests_too: bool = True) -> tuple[list[Finding], CheckContext]:
    ctx = CheckContext(project=project, workspace=ws)
    copy = ws.fresh_copy("baseline")
    findings = check_dependencies(ctx)
    findings += check_imports(ctx, copy)
    if run_tests_too:
        findings += check_tests(ctx, copy, test_command)
    findings += check_saved_agents(ctx, agent_paths)
    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order[f.severity], f.check, f.file or "", f.line or 0))
    return findings, ctx
