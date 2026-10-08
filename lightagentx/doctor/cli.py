"""`lightx doctor` — the command-line front end."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .checks import default_test_command, run_checks
from .project import Finding, Workspace

_C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[91m",
      "yellow": "\033[93m", "green": "\033[92m", "cyan": "\033[96m"}
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    _C = {k: "" for k in _C}

_ICON = {"error": ("✗", "red"), "warning": ("!", "yellow"), "info": ("·", "dim")}
_TITLES = {"dependency": "Dependencies", "import": "Imports", "deprecation": "Deprecations",
           "saved-agent": "Saved agents", "tests": "Tests"}


def render(findings: list[Finding], project: Path) -> str:
    out = [f"{_C['bold']}lightx doctor{_C['reset']}  {project}"]
    for check in _TITLES:
        group = [f for f in findings if f.check == check]
        if not group:
            continue
        out.append(f"\n{_C['cyan']}{_C['bold']}{_TITLES[check]}{_C['reset']}")
        for f in group:
            icon, color = _ICON[f.severity]
            where = f"  {_C['dim']}{f.where}{_C['reset']}" if f.where else ""
            fix = f"  {_C['green']}[fixable]{_C['reset']}" if f.fixable else ""
            out.append(f"  {_C[color]}{icon}{_C['reset']} {f.title}{where}{fix}")
            for line in f.detail.splitlines()[:6]:
                out.append(f"      {_C['dim']}{line}{_C['reset']}")
    errors = sum(f.severity == "error" for f in findings)
    warnings = sum(f.severity == "warning" for f in findings)
    fixable = sum(bool(f.fixable) for f in findings)
    if errors == warnings == 0:
        out.append(f"\n{_C['green']}No problems found.{_C['reset']}")
    else:
        out.append(f"\n{errors} error(s), {warnings} warning(s), {fixable} fixable. "
                   + ("Run `lightx doctor --fix` to repair them (each change asks you first)." if fixable else ""))
    return "\n".join(out)


def _approve(question: str) -> bool:
    if not sys.stdin.isatty():
        print(f"  {question} — no terminal attached, so not applied.")
        return False
    try:
        return input(f"  {_C['yellow']}{question} [y/N] {_C['reset']}").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", nargs="?", default=".", help="Project to check (default: current directory).")
    p.add_argument("--fix", action="store_true",
                   help="Propose fixes, verify each in the sandbox, and apply only with your approval.")
    p.add_argument("--json", action="store_true", help="Print findings as JSON.")
    p.add_argument("--agents", action="append", default=[], metavar="PATH",
                   help="Extra snapshot/archive file or registry folder to check (repeatable).")
    p.add_argument("--test-cmd", help="Test command to run (default: pytest if tests exist).")
    p.add_argument("--no-tests", action="store_true", help="Don't run the project's tests.")
    p.add_argument("--test-timeout", type=float, default=300, help="Seconds (default 300).")
    p.add_argument("--max-attempts", type=int, default=2, help="LLM attempts per problem (default 2).")
    p.add_argument("--only", default="code,dependency,migration",
                   help="Kinds of fixes to attempt, comma-separated: code, dependency, migration.")
    p.add_argument("--backend", default="auto", choices=["auto", "bubblewrap", "subprocess"],
                   help="Sandbox backend for running the project's code.")
    p.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic", "gemini"])
    p.add_argument("--model")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1")


def run(args: argparse.Namespace) -> int:
    project = Path(args.path).expanduser().resolve()
    if not project.is_dir():
        print(f"Not a directory: {project}", file=sys.stderr)
        return 2
    test_cmd = None if args.no_tests else (
        shlex.split(args.test_cmd) if args.test_cmd else default_test_command(project))
    ws = Workspace(project, backend=args.backend, timeout_s=args.test_timeout)
    try:
        if not args.json:
            print(f"{_C['dim']}Checking (your project is copied; its code runs only inside the sandbox)…{_C['reset']}")
        findings, ctx = run_checks(project, ws=ws, test_command=test_cmd,
                                   agent_paths=[Path(a).expanduser().resolve() for a in args.agents],
                                   run_tests_too=not args.no_tests)
        if args.json and not args.fix:
            print(json.dumps([f.to_dict() for f in findings], indent=2))
            return 1 if any(f.severity == "error" for f in findings) else 0
        print(render(findings, project))
        if not args.fix or not any(f.fixable for f in findings):
            return 1 if any(f.severity == "error" for f in findings) else 0

        llm = None
        if any(f.fixable == "code" for f in findings) and "code" in args.only:
            from ..llm.auto import NoLLMConfigured, llm_from_env
            try:
                llm = llm_from_env(args.provider, args.model, args.base_url, temperature=0.0)
            except NoLLMConfigured as e:
                print(f"\n{e}")
        from .fixer import run_fixes
        kinds = {k.strip() for k in args.only.split(",") if k.strip()}
        unknown = kinds - {"code", "dependency", "migration"}
        if unknown:
            print(f"Unknown --only kind(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        findings = [f for f in findings if f.fixable in kinds]
        log = run_fixes(findings, ctx, llm=llm, test_command=test_cmd, approve=_approve,
                        say=print, max_attempts=args.max_attempts)
        # Fixes can reveal problems that were hidden (e.g. a deprecation that only
        # shows once the tests can run). Re-check and offer another round.
        attempted = {(e["finding"], e["where"]) for e in log}
        for _ in range(2):
            if not any(e["outcome"] == "applied" for e in log):
                break
            print(f"\n{_C['dim']}Re-checking after the applied fixes…{_C['reset']}")
            again, ctx = run_checks(project, ws=ws, test_command=test_cmd,
                                    agent_paths=[Path(a).expanduser().resolve() for a in args.agents],
                                    run_tests_too=not args.no_tests)
            new = [f for f in again if f.fixable in kinds and (f.title, f.where) not in attempted]
            if not new:
                break
            print(render(new, project))
            more = run_fixes(new, ctx, llm=llm, test_command=test_cmd, approve=_approve,
                             say=print, max_attempts=args.max_attempts)
            attempted |= {(e["finding"], e["where"]) for e in more}
            log += more
            if not any(e["outcome"] == "applied" for e in more):
                break
        print(f"\n{_C['bold']}Summary{_C['reset']}")
        for e in log:
            color = {"applied": "green", "already fixed": "green"}.get(e["outcome"], "yellow")
            print(f"  {_C[color]}{e['outcome']:<15}{_C['reset']} {e['finding']}"
                  + (f"  {_C['dim']}{e['where']}{_C['reset']}" if e["where"] else ""))
        return 0
    finally:
        ws.cleanup()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lightx doctor",
                                description="Find version problems; fix them with verified, approved patches.")
    add_arguments(p)
    return run(p.parse_args(argv))
