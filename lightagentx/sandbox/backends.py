"""Sandbox Backends — the actual isolation mechanisms that run agent commands."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .policy import SandboxPolicy


@dataclass
class ExecResult:
    """Outcome of one sandboxed command."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    duration_s: float = 0.0

    def to_text(self, max_chars: int = 20_000) -> str:
        """Compact text form for handing back to an LLM."""
        parts = [f"exit_code={self.exit_code}" + (" (TIMED OUT)" if self.timed_out else "")]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
        return text


# Only these variables reach sandboxed commands — API keys and tokens never do.
_SAFE_ENV_VARS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "TZ")


def _safe_env(policy: SandboxPolicy) -> dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV_VARS if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = str(policy.workspace)
    env["TMPDIR"] = "/tmp" if os.name == "posix" else str(policy.workspace)
    env["LIGHTX_SANDBOX"] = "1"
    return env


def _rlimit_preexec(policy: SandboxPolicy):
    """Build a preexec_fn that applies resource limits (POSIX only)."""
    if os.name != "posix":
        return None

    import resource

    mem = policy.memory_mb * 1024 * 1024
    cpu = int(policy.timeout_s) + 5

    def apply() -> None:
        for limit, value in (
            (resource.RLIMIT_AS, mem),
            (resource.RLIMIT_CPU, cpu),
            (resource.RLIMIT_FSIZE, 512 * 1024 * 1024),
            (resource.RLIMIT_CORE, 0),
        ):
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass

    return apply


class SandboxBackend(ABC):
    """
    Contract for an isolation mechanism.

    ``isolated`` tells the guard whether commands are truly contained.
    Unisolated backends make every command HIGH risk (needs confirmation).
    """

    name: str = "base"
    isolated: bool = False

    @classmethod
    def available(cls) -> bool:
        return True

    @abstractmethod
    def build_argv(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        """Wrap a command in this backend's isolation."""
        ...

    def run(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        stdin: str | None = None,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a command under this backend with the policy's limits.

        `env` adds explicit variables (e.g. PYTHONPATH) on top of the scrubbed
        environment; nothing else from the caller's environment passes through.
        """
        policy.workspace.mkdir(parents=True, exist_ok=True)
        timeout = min(timeout_s or policy.timeout_s, policy.timeout_s)
        full_argv = self.build_argv(argv, policy)

        start = time.monotonic()
        proc = subprocess.Popen(
            full_argv,
            cwd=str(policy.workspace),
            env={**_safe_env(policy), **(env or {})},
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_rlimit_preexec(policy),
            start_new_session=(os.name == "posix"),
        )
        timed_out = False
        try:
            out, err = proc.communicate(
                input=stdin.encode() if stdin is not None else None, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            out, err = proc.communicate()

        return ExecResult(
            stdout=out.decode(errors="replace").rstrip(),
            stderr=err.decode(errors="replace").rstrip(),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            duration_s=round(time.monotonic() - start, 3),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(isolated={self.isolated})"


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


class SubprocessBackend(SandboxBackend):
    """
    Works on every OS. Gives a private working directory, a scrubbed
    environment, resource limits and a timeout — but NOT filesystem or
    network isolation. Treat commands as running with your user's rights.
    """

    name = "subprocess"
    isolated = False

    def build_argv(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        return list(argv)


class BubblewrapBackend(SandboxBackend):
    """
    Linux namespaces via bubblewrap (``bwrap``), no root required.

    Inside the sandbox:
      - the whole filesystem is read-only
      - $HOME is empty except the policy's read_paths (read-only)
      - deny_paths are hidden, /tmp and /run are private
      - only the workspace is writable
      - no network (unless allow_network), private PID/IPC/UTS namespaces
      - the session D-Bus / Wayland / X11 sockets are unreachable
    """

    name = "bubblewrap"
    isolated = True
    _available: bool | None = None

    @classmethod
    def available(cls) -> bool:
        if cls._available is None:
            cls._available = False
            if sys.platform.startswith("linux") and shutil.which("bwrap"):
                try:
                    r = subprocess.run(
                        ["bwrap", "--ro-bind", "/", "/", "--unshare-all", "true"],
                        capture_output=True, timeout=10,
                    )
                    cls._available = r.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    pass
        return cls._available

    def build_argv(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        home = Path.home().resolve()
        ws = str(policy.workspace)

        args = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--hostname", "lightx-sandbox",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
            "--tmpfs", str(home),
        ]
        if policy.allow_network:
            args += ["--share-net"]
            # Keep DNS working on systemd-resolved hosts.
            args += ["--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve"]

        # $HOME, /tmp and /run are replaced by empty tmpfs above; re-expose the
        # allowed read paths that live under them (read-only).
        hidden_roots = [home, Path("/tmp"), Path("/run")]
        for p in policy.read_paths:
            if any(p == r or r in p.parents for r in hidden_roots):
                args += ["--ro-bind-try", str(p), str(p)]

        for p in policy.deny_paths:
            if p.is_dir():
                args += ["--tmpfs", str(p)]
            elif p.exists():
                args += ["--ro-bind", "/dev/null", str(p)]

        args += ["--bind", ws, ws, "--chdir", ws, "--"]
        return args + list(argv)


def auto_backend() -> SandboxBackend:
    """Pick the strongest backend available on this machine."""
    if BubblewrapBackend.available():
        return BubblewrapBackend()
    return SubprocessBackend()
