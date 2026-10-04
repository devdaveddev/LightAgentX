"""Sandbox Policy — declarative rules for what agents may touch on the host OS."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Risk(enum.IntEnum):
    """
    How dangerous an action is. Actions at or above the policy's
    ``confirm_at`` level need a human "yes" before they run.

        LOW     — read-only (list files, show processes, scan ports)
        MEDIUM  — reversible or contained (write in workspace, isolated command)
        HIGH    — destructive or outside the sandbox (kill, delete, unisolated command)
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


def _home() -> Path:
    return Path.home()


def _default_deny_paths() -> list[Path]:
    """Secrets and credential stores agents must never read or write."""
    home = _home()
    rel = [
        ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker",
        ".config/gcloud", ".config/gh", ".netrc", ".pgpass", ".git-credentials",
        ".password-store", ".local/share/keyrings", ".pki",
        ".mozilla", ".config/google-chrome", ".config/chromium",
        ".config/BraveSoftware", ".lightx/audit.jsonl", ".lightx/config.json",
    ]
    paths = [home / r for r in rel]
    paths += [Path("/etc/shadow"), Path("/etc/gshadow"), Path("/etc/sudoers"), Path("/root")]
    return paths


_DEFAULT_DENY_GLOBS = [
    "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*",
    ".env", ".env.*", "*.kdbx",
]

# Defense in depth on top of isolation: commands that are never worth running.
_DEFAULT_BLOCKED_COMMANDS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]*\s+)*(/|~|\$HOME)\s*($|[;&|])",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b.*\bof=/dev/",
    r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b",
    r"\bsudo\b|\bsu\s|\bdoas\b|\bpkexec\b",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bchmod\s+(-R\s+)?[0-7]*777\s+/",
    r">\s*/dev/sd[a-z]",
    r"\b(curl|wget)\b[^|]*\|\s*(ba|z|)sh\b",
]

_DEFAULT_PROTECTED_PROCESSES = [
    "systemd", "init", "launchd", "kernel_task", "wininit.exe", "csrss.exe",
    "lsass.exe", "services.exe", "smss.exe", "explorer.exe",
    "sshd", "dbus-daemon", "dbus-broker", "Xorg", "Xwayland", "gnome-shell",
    "kwin_wayland", "kwin_x11", "plasmashell", "gdm", "sddm", "lightdm",
    "pipewire", "wireplumber", "pulseaudio", "NetworkManager", "polkitd",
    "login", "agetty", "firewalld", "auditd", "WindowServer", "loginwindow",
]


@dataclass
class SandboxPolicy:
    """
    Everything the sandbox enforces, in one place.

    Attributes:
        workspace: The agents' private scratch directory (read-write, and the
            only writable place for isolated commands).
        read_paths: Directories file tools may read from.
        write_paths: Directories file tools may write to (workspace is always included).
        deny_paths: Never readable or writable, even if inside read/write paths.
        deny_globs: Filename patterns that are never readable or writable.
        allow_network: Whether sandboxed commands get network access.
        timeout_s: Wall-clock limit for one sandboxed command.
        memory_mb: Address-space limit for sandboxed commands.
        max_output_chars: Truncate tool output beyond this (protects the context window).
        blocked_commands: Regexes; matching shell commands are refused outright.
        protected_processes: Process names that may never be terminated.
        trusted_apps: App ids/names that can be launched without confirmation.
        confirm_at: Minimum Risk level that requires human confirmation.
    """

    workspace: Path = field(default_factory=lambda: _home() / ".lightx" / "workspace")
    read_paths: list[Path] = field(default_factory=lambda: [_home()])
    write_paths: list[Path] = field(default_factory=list)
    deny_paths: list[Path] = field(default_factory=_default_deny_paths)
    deny_globs: list[str] = field(default_factory=lambda: list(_DEFAULT_DENY_GLOBS))
    allow_network: bool = False
    timeout_s: float = 30.0
    memory_mb: int = 1024
    max_output_chars: int = 20_000
    blocked_commands: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKED_COMMANDS))
    protected_processes: list[str] = field(default_factory=lambda: list(_DEFAULT_PROTECTED_PROCESSES))
    trusted_apps: list[str] = field(default_factory=list)
    confirm_at: Risk = Risk.HIGH

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.read_paths = [Path(p).expanduser().resolve() for p in self.read_paths]
        self.write_paths = [Path(p).expanduser().resolve() for p in self.write_paths]
        self.deny_paths = [Path(p).expanduser().resolve() for p in self.deny_paths]
        if self.workspace not in self.write_paths:
            self.write_paths.insert(0, self.workspace)
        if self.workspace not in self.read_paths:
            self.read_paths.insert(0, self.workspace)
        self.confirm_at = Risk(self.confirm_at)
        self._blocked_re = [re.compile(p) for p in self.blocked_commands]

    # ── Checks ────────────────────────────────────────────────────────────

    def is_denied(self, path: Path) -> bool:
        """True if the path is a secret/credential location."""
        for denied in self.deny_paths:
            if path == denied or denied in path.parents:
                return True
        return any(path.match(g) for g in self.deny_globs)

    def can_read(self, path: Path) -> bool:
        return not self.is_denied(path) and _under_any(path, self.read_paths)

    def can_write(self, path: Path) -> bool:
        return not self.is_denied(path) and _under_any(path, self.write_paths)

    def blocked_command_reason(self, command: str) -> str | None:
        """Return the matching pattern if a command is blocked, else None."""
        for rx in self._blocked_re:
            if rx.search(command):
                return rx.pattern
        return None

    def is_protected_process(self, name: str) -> bool:
        return name in self.protected_processes

    def needs_confirmation(self, risk: Risk) -> bool:
        return risk >= self.confirm_at

    # ── Serialization (travels with agent snapshots) ──────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "read_paths": [str(p) for p in self.read_paths],
            "write_paths": [str(p) for p in self.write_paths],
            "deny_paths": [str(p) for p in self.deny_paths],
            "deny_globs": list(self.deny_globs),
            "allow_network": self.allow_network,
            "timeout_s": self.timeout_s,
            "memory_mb": self.memory_mb,
            "max_output_chars": self.max_output_chars,
            "blocked_commands": list(self.blocked_commands),
            "protected_processes": list(self.protected_processes),
            "trusted_apps": list(self.trusted_apps),
            "confirm_at": self.confirm_at.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxPolicy":
        data = dict(data)
        if isinstance(data.get("confirm_at"), str):
            data["confirm_at"] = Risk[data["confirm_at"]]
        return cls(**data)


def _under_any(path: Path, roots: list[Path]) -> bool:
    return any(path == r or r in path.parents for r in roots)
