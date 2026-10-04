"""lightx-os — chat with your computer through sandboxed agents."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..hooks import HookEvent, HookRegistry
from ..llm.base import BaseLLM
from ..sandbox import Risk, Sandbox, SandboxPolicy, SandboxViolation, SubprocessBackend, auto_backend

_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[91m",
    "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m", "magenta": "\033[95m",
}
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    _C = {k: "" for k in _C}

_HELP = """Talk naturally, e.g.:
  what's using my CPU?            close spotify
  find my resume pdf              open ~/Downloads/report.pdf
  run a security scan             is anything listening on the network?
  launch firefox                  show disk usage of my home folder

Direct commands (no LLM):
  /stats          system overview           /ps [cpu|memory] [N]   process list
  /kill PID       terminate a process       /scan                  security scan
  /apps [query]   installed applications    /policy                sandbox rules
  /audit [N]      recent sandbox decisions  /reset                 forget conversation
  /help           this help                 /exit                  quit"""


def make_llm(provider: str, model: str | None, base_url: str | None) -> BaseLLM:
    if provider == "auto":
        if base_url:
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            raise SystemExit(
                "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY, "
                "or point --base-url at an OpenAI-compatible local server (e.g. Ollama)."
            )
    kwargs = {"temperature": 0.2, "max_tokens": 2048}
    if model:
        kwargs["model"] = model
    if provider == "anthropic":
        from ..llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(**kwargs)
    if provider == "gemini":
        from ..llm.gemini_llm import GeminiLLM
        return GeminiLLM(**kwargs)
    from ..llm.openai_llm import OpenAILLM
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "local")
    return OpenAILLM(**kwargs)


def terminal_confirmer(description: str, risk: Risk) -> bool:
    """Ask the human at the keyboard. Non-interactive sessions always decline."""
    if not sys.stdin.isatty():
        return False
    color = _C["red"] if risk >= Risk.HIGH else _C["yellow"]
    print(f"\n{color}{_C['bold']}⚠ {risk.name} RISK{_C['reset']}  {description}")
    try:
        answer = input(f"{color}Allow? [y/N] {_C['reset']}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _trace_hooks(show: bool) -> HookRegistry:
    hooks = HookRegistry()
    if not show:
        return hooks

    @hooks.on("on_route")
    def _route(e: HookEvent) -> None:
        names = ", ".join(d.get("agent", "?") for d in e.data["plan"])
        print(f"{_C['dim']}  ↳ {names}{_C['reset']}")

    @hooks.on("before_tool_call")
    def _tool(e: HookEvent) -> None:
        args = ", ".join(f"{k}={v!r}"[:60] for k, v in e.data["arguments"].items())
        print(f"{_C['dim']}  · {e.data['tool_name']}({args}){_C['reset']}")

    return hooks


def build_sandbox(args: argparse.Namespace) -> Sandbox:
    policy = SandboxPolicy(
        workspace=Path(args.workspace) if args.workspace else Path.home() / ".lightx" / "workspace",
        write_paths=[Path(p) for p in args.allow_write],
        allow_network=args.network,
        confirm_at=Risk[args.confirm.upper()],
        trusted_apps=args.trust_app,
    )
    backend = SubprocessBackend() if args.no_isolation else auto_backend()
    return Sandbox(
        policy=policy, backend=backend, confirmer=terminal_confirmer,
        audit_file=Path.home() / ".lightx" / "audit.jsonl",
    )


def _direct_command(line: str, sandbox: Sandbox, smart_os) -> bool:
    """Handle /commands. Returns False when the user wants to quit."""
    from . import tasks
    from .apps import discover_apps
    from .security import make_security_tools

    cmd, *rest = line[1:].split()
    try:
        if cmd in ("exit", "quit", "q"):
            return False
        elif cmd == "help":
            print(_HELP)
        elif cmd == "stats":
            print(tasks.system_overview_text())
        elif cmd == "ps":
            sort = rest[0] if rest and not rest[0].isdigit() else "cpu"
            n = int(next((r for r in rest if r.isdigit()), 15))
            print(tasks.format_processes(tasks.snapshot_processes(sort, n)))
        elif cmd == "kill":
            if not rest or not rest[0].isdigit():
                print("usage: /kill PID")
            else:
                print(tasks.terminate_pid(sandbox, int(rest[0]), force="-9" in rest))
        elif cmd == "scan":
            print(next(t for t in make_security_tools(sandbox) if t.name == "security_scan")())
        elif cmd == "apps":
            q = " ".join(rest).lower()
            for a in sorted(discover_apps().values(), key=lambda a: a.name.lower()):
                if not q or q in a.name.lower() or q in a.app_id.lower():
                    print(f"  {a.name}  {_C['dim']}[{a.app_id}]{_C['reset']}")
        elif cmd == "policy":
            print(sandbox.describe())
        elif cmd == "audit":
            n = int(rest[0]) if rest and rest[0].isdigit() else 15
            for e in sandbox.audit_log[-n:]:
                print(f"  {e.decision:<9} {e.risk:<6} {e.action:<12} {e.target[:80]}")
            if not sandbox.audit_log:
                print("  (no sandbox decisions yet this session)")
        elif cmd == "reset":
            smart_os.reset()
            print("Conversation cleared.")
        else:
            print(f"Unknown command /{cmd}. Try /help.")
    except SandboxViolation as e:
        print(f"{_C['red']}✗ {e}{_C['reset']}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lightx-os", description="Chat with your computer through sandboxed AI agents.",
    )
    parser.add_argument("-c", "--command", help="Run one request and exit.")
    parser.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic", "gemini"])
    parser.add_argument("--model", help="Model name for the chosen provider.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (e.g. http://localhost:11434/v1 for Ollama).")
    parser.add_argument("--mode", default="crew", choices=["crew", "single"],
                        help="crew = router + specialist agents; single = one agent with all tools.")
    parser.add_argument("--workspace", help="Agent workspace directory (default ~/.lightx/workspace).")
    parser.add_argument("--allow-write", action="append", default=[], metavar="PATH",
                        help="Extra directory file tools may write to (repeatable).")
    parser.add_argument("--network", action="store_true", help="Give sandboxed commands network access.")
    parser.add_argument("--confirm", default="high", choices=["low", "medium", "high"],
                        help="Ask before actions at this risk level and above (default: high).")
    parser.add_argument("--trust-app", action="append", default=[], metavar="APP",
                        help="App that may launch without confirmation (repeatable).")
    parser.add_argument("--no-isolation", action="store_true",
                        help="Force the plain subprocess backend (commands then always need confirmation).")
    parser.add_argument("--quiet", action="store_true", help="Hide routing and tool-call traces.")
    parser.add_argument("--verbose", action="store_true", help="Show full agent reasoning logs.")
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument("--enable", action="store_true", help="Turn SmartOS on.")
    switch.add_argument("--disable", action="store_true",
                        help="Turn SmartOS off; LightAgentX stays a plain agent framework.")
    switch.add_argument("--status", action="store_true", help="Show whether SmartOS is on.")
    args = parser.parse_args(argv)

    if args.enable or args.disable or args.status:
        from ..features import launch_smartos
        return launch_smartos(["--enable" if args.enable else "--disable" if args.disable else "--status"])

    from .os_agent import SmartOS

    sandbox = build_sandbox(args)
    llm = make_llm(args.provider, args.model, args.base_url)
    smart_os = SmartOS(llm=llm, sandbox=sandbox, mode=args.mode,
                       hooks=_trace_hooks(not args.quiet), verbose=args.verbose)

    if args.command:
        print(smart_os.run(args.command))
        return 0

    iso = (f"{_C['green']}isolated ({sandbox.backend.name}){_C['reset']}" if sandbox.backend.isolated
           else f"{_C['yellow']}NOT isolated — every command needs your approval{_C['reset']}")
    print(f"{_C['cyan']}{_C['bold']}LightX SmartOS{_C['reset']}  ·  {llm}  ·  {args.mode} mode  ·  sandbox {iso}")
    print(f"{_C['dim']}Type /help for examples and commands, /exit to quit.{_C['reset']}")

    while True:
        try:
            line = input(f"\n{_C['magenta']}{_C['bold']}you ›{_C['reset']} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not _direct_command(line, sandbox, smart_os):
                break
            continue
        try:
            answer = smart_os.run(line)
        except KeyboardInterrupt:
            print(f"\n{_C['yellow']}Interrupted.{_C['reset']}")
            continue
        except Exception as e:  # keep the session alive on provider/network errors
            print(f"{_C['red']}Error: {type(e).__name__}: {e}{_C['reset']}")
            continue
        print(f"\n{_C['cyan']}{_C['bold']}os ›{_C['reset']} {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
