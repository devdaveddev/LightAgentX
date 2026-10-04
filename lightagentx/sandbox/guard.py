"""Sandbox — the single gatekeeper every OS action passes through."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .backends import ExecResult, SandboxBackend, auto_backend
from .policy import Risk, SandboxPolicy


class SandboxViolation(PermissionError):
    """Raised when an action is outside the policy or the human declines it.

    The ToolExecutor turns this into an error string the LLM can read, so the
    agent learns *why* it was stopped instead of crashing.
    """


Confirmer = Callable[[str, Risk], bool]


def deny_all(description: str, risk: Risk) -> bool:
    """Default confirmer: no human attached, so nothing risky runs."""
    return False


def allow_all(description: str, risk: Risk) -> bool:
    """For tests and fully trusted automation only."""
    return True


@dataclass
class AuditEntry:
    timestamp: float
    action: str
    target: str
    risk: str
    decision: str
    detail: str = ""


@dataclass
class Sandbox:
    """
    Combines a policy, an isolation backend, a human-confirmation callback
    and an audit trail.

    Usage::

        sandbox = Sandbox()                       # auto-picks bubblewrap on Linux
        sandbox.check_read("~/notes.txt")         # -> resolved Path or SandboxViolation
        sandbox.authorize("terminate", "pid 42", Risk.HIGH)
        result = sandbox.run_shell("ls -la")
    """

    policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    backend: SandboxBackend = field(default_factory=auto_backend)
    confirmer: Confirmer = deny_all
    audit_file: Path | None = None
    audit_log: list[AuditEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.policy.workspace.mkdir(parents=True, exist_ok=True)
        if self.audit_file is not None:
            self.audit_file = Path(self.audit_file).expanduser()
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────────────────

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path; relative paths are relative to the workspace."""
        p = Path(os.path.expandvars(str(path))).expanduser()
        if not p.is_absolute():
            p = self.policy.workspace / p
        return p.resolve()

    def check_read(self, path: str | Path) -> Path:
        p = self.resolve(path)
        if not self.policy.can_read(p):
            self._record("read", str(p), Risk.LOW, "blocked")
            raise SandboxViolation(f"Read access to '{p}' is not allowed by the sandbox policy.")
        return p

    def check_write(self, path: str | Path) -> Path:
        p = self.resolve(path)
        if not self.policy.can_write(p):
            self._record("write", str(p), Risk.MEDIUM, "blocked")
            raise SandboxViolation(
                f"Write access to '{p}' is not allowed. Writable locations: "
                f"{[str(w) for w in self.policy.write_paths]}"
            )
        return p

    def in_workspace(self, path: Path) -> bool:
        ws = self.policy.workspace
        return path == ws or ws in path.parents

    # ── Authorization ─────────────────────────────────────────────────────

    def authorize(self, action: str, target: str, risk: Risk, detail: str = "") -> None:
        """
        Gate an action. Low-risk actions pass; risky ones go to the human.
        Raises SandboxViolation if declined.
        """
        if self.policy.needs_confirmation(risk):
            description = f"{action}: {target}" + (f"\n    {detail}" if detail else "")
            if not self.confirmer(description, risk):
                self._record(action, target, risk, "declined", detail)
                raise SandboxViolation(
                    f"The user declined '{action}' on '{target}'. "
                    f"Do not retry; ask the user how they want to proceed."
                )
            self._record(action, target, risk, "approved", detail)
        else:
            self._record(action, target, risk, "allowed", detail)

    def violation(self, action: str, target: str, reason: str,
                  risk: Risk = Risk.HIGH) -> SandboxViolation:
        """Record a policy refusal in the audit log and return the exception to raise."""
        self._record(action, target, risk, "blocked", reason)
        return SandboxViolation(reason)

    # ── Command execution ─────────────────────────────────────────────────

    @property
    def command_risk(self) -> Risk:
        return Risk.MEDIUM if self.backend.isolated else Risk.HIGH

    def run_shell(self, command: str, timeout_s: float | None = None) -> ExecResult:
        """Run a shell command inside the sandbox backend."""
        reason = self.policy.blocked_command_reason(command)
        if reason:
            self._record("run_command", command, Risk.HIGH, "blocked", reason)
            raise SandboxViolation(f"Command blocked by policy (matched {reason!r}).")

        self.authorize("run_command", command, self.command_risk,
                       f"backend={self.backend.name}")
        argv = ["/bin/sh", "-c", command] if os.name == "posix" else ["cmd", "/c", command]
        return self.backend.run(argv, self.policy, timeout_s=timeout_s)

    def run_argv(self, argv: list[str], stdin: str | None = None,
                 timeout_s: float | None = None) -> ExecResult:
        """Run a fixed argv inside the sandbox (caller must authorize)."""
        return self.backend.run(argv, self.policy, stdin=stdin, timeout_s=timeout_s)

    # ── Audit ─────────────────────────────────────────────────────────────

    def _record(self, action: str, target: str, risk: Risk, decision: str,
                detail: str = "") -> None:
        entry = AuditEntry(time.time(), action, target[:500], risk.name, decision, detail[:500])
        self.audit_log.append(entry)
        if self.audit_file is not None:
            try:
                with self.audit_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
            except OSError:
                pass

    def truncate(self, text: str) -> str:
        limit = self.policy.max_output_chars
        if len(text) > limit:
            return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
        return text

    def describe(self) -> str:
        """Human/LLM-readable summary of the active sandbox."""
        p = self.policy
        return (
            f"backend: {self.backend.name} (isolated={self.backend.isolated})\n"
            f"workspace: {p.workspace}\n"
            f"readable: {', '.join(map(str, p.read_paths))}\n"
            f"writable: {', '.join(map(str, p.write_paths))}\n"
            f"network for commands: {'on' if p.allow_network else 'off'}\n"
            f"timeout: {p.timeout_s}s, memory: {p.memory_mb}MB\n"
            f"confirmation required at: {p.confirm_at.name} risk and above"
        )
