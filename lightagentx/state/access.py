"""Access control and redaction for shared agent state."""

from __future__ import annotations

import copy
import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

# Rights, from least to most powerful. "private" lets a principal see private
# state keys and unredacted transcripts; everything else sees a redacted view.
RIGHTS = ("read", "run", "write", "fork", "merge", "private", "admin")

_IMPLIES = {
    "run": {"read"},
    "write": {"read", "run"},
    "fork": {"read"},
    "merge": {"read"},
    "admin": set(RIGHTS),
}


class AccessDenied(PermissionError):
    pass


def expand(rights: set[str]) -> set[str]:
    out = set(rights)
    for r in rights:
        out |= _IMPLIES.get(r, set())
    return out


# Secrets that must never leave the store in a redacted view.
_SECRET_PATTERNS = [
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{16,}"),            # OpenAI / Anthropic keys
    re.compile(r"AKIA[0-9A-Z]{16}"),                            # AWS access key id
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),                  # GitHub tokens
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),                      # Google API keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(password|passwd|secret|token)\s*[:=]\s*\S+"),
]

REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    for rx in _SECRET_PATTERNS:
        text = rx.sub(REDACTED, text)
    return text


@dataclass
class AccessPolicy:
    """
    Who may do what with one agent.

    Attributes:
        owner: Principal with every right.
        grants: principal (or glob like "billing-*") -> set of rights.
        private_keys: Top-level state keys only "private" principals can see.
        write_branches: principal glob -> branch globs it may commit to
            (default: any branch for principals with "write").
    """

    owner: str
    grants: dict[str, set[str]] = field(default_factory=dict)
    private_keys: set[str] = field(default_factory=set)
    write_branches: dict[str, list[str]] = field(default_factory=dict)

    def rights_of(self, principal: str) -> set[str]:
        if principal == self.owner:
            return set(RIGHTS)
        rights: set[str] = set()
        for pattern, granted in self.grants.items():
            if fnmatch.fnmatchcase(principal, pattern):
                rights |= set(granted)
        return expand(rights)

    def can(self, principal: str, right: str) -> bool:
        return right in self.rights_of(principal)

    def can_write_branch(self, principal: str, branch: str) -> bool:
        if not self.can(principal, "write"):
            return False
        if principal == self.owner:
            return True
        for pattern, branches in self.write_branches.items():
            if fnmatch.fnmatchcase(principal, pattern):
                return any(fnmatch.fnmatchcase(branch, b) for b in branches)
        return True

    # ── redacted views ────────────────────────────────────────────────────

    def view_state(self, principal: str, state: dict[str, Any]) -> dict[str, Any]:
        if self.can(principal, "private"):
            return copy.deepcopy(state)
        return {k: _redact_value(v) for k, v in state.items() if k not in self.private_keys}

    def view_transcript(self, principal: str, transcript: list[dict]) -> list[dict]:
        if self.can(principal, "private"):
            return copy.deepcopy(list(transcript))
        out = []
        for m in transcript:
            m = copy.deepcopy(m)
            if isinstance(m.get("content"), str):
                m["content"] = redact_text(m["content"])
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = redact_text(fn["arguments"])
            out.append(m)
        return out

    # ── persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "grants": {k: sorted(v) for k, v in self.grants.items()},
            "private_keys": sorted(self.private_keys),
            "write_branches": self.write_branches,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccessPolicy":
        return cls(
            owner=d["owner"],
            grants={k: set(v) for k, v in d.get("grants", {}).items()},
            private_keys=set(d.get("private_keys", [])),
            write_branches=d.get("write_branches", {}),
        )


def _redact_value(v: Any) -> Any:
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, list):
        return [_redact_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _redact_value(x) for k, x in v.items()}
    return v
