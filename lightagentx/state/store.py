"""Content-addressed version store with locked, compare-and-swap branch refs."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BRANCH_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class ConflictError(RuntimeError):
    """A branch moved since the session attached (compare-and-swap failed)."""

    def __init__(self, branch: str, expected: str | None, actual: str | None):
        self.branch, self.expected, self.actual = branch, expected, actual
        super().__init__(
            f"Branch '{branch}' moved: expected {_short(expected)}, found {_short(actual)}."
        )


class IntegrityError(RuntimeError):
    """A stored version does not match its content hash."""


def _short(vid: str | None) -> str:
    return vid[:12] if vid else "(none)"


def canonical_json(data: Any) -> str:
    """Deterministic JSON — the input to content hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_id(content: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def current_provenance(principal: str, workflow: str = "") -> dict[str, Any]:
    """Who/what/where produced a version. Asserted, not cryptographically signed."""
    try:
        os_user = getpass.getuser()
    except Exception:  # pragma: no cover - no login name in some containers
        os_user = "?"
    return {
        "principal": principal,
        "workflow": workflow,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "os_user": os_user,
        "program": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "python",
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class Version:
    """An immutable, content-addressed agent state version (like a git commit)."""

    id: str
    agent_id: str
    parents: tuple[str, ...]
    created_at: float
    message: str
    provenance: dict[str, Any]
    config: dict[str, Any]
    transcript: tuple[dict[str, Any], ...]
    state: dict[str, Any]
    tool_manifest: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def content_of(**fields: Any) -> dict[str, Any]:
        return {
            "agent_id": fields["agent_id"],
            "parents": list(fields["parents"]),
            "created_at": fields["created_at"],
            "message": fields["message"],
            "provenance": fields["provenance"],
            "config": fields["config"],
            "transcript": list(fields["transcript"]),
            "state": fields["state"],
            "tool_manifest": list(fields.get("tool_manifest", ())),
        }

    @classmethod
    def create(cls, **fields: Any) -> "Version":
        content = cls.content_of(**fields)
        # Round-trip through JSON so the in-memory object equals what is stored.
        content = json.loads(canonical_json(content))
        return cls.from_content(content_id(content), content)

    @classmethod
    def from_content(cls, vid: str, c: dict[str, Any]) -> "Version":
        return cls(
            id=vid, agent_id=c["agent_id"], parents=tuple(c["parents"]),
            created_at=c["created_at"], message=c["message"], provenance=c["provenance"],
            config=c["config"], transcript=tuple(c["transcript"]), state=c["state"],
            tool_manifest=tuple(c.get("tool_manifest", [])),
        )

    def content(self) -> dict[str, Any]:
        return self.content_of(**{f: getattr(self, f) for f in (
            "agent_id", "parents", "created_at", "message", "provenance",
            "config", "transcript", "state", "tool_manifest")})

    @property
    def short(self) -> str:
        return self.id[:12]


class FileLock:
    """Cross-process exclusive lock on a file (fcntl on POSIX, msvcrt on Windows)."""

    def __init__(self, path: Path, timeout_s: float = 30.0):
        self.path = path
        self.timeout_s = timeout_s
        self._fh = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                if os.name == "nt":  # pragma: no cover
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > deadline:
                    self._fh.close()
                    raise TimeoutError(f"Could not lock {self.path} within {self.timeout_s}s")
                time.sleep(0.005)

    def __exit__(self, *exc: Any) -> None:
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class VersionStore:
    """
    On-disk layout::

        root/objects/ab/cdef...json      immutable versions, shared by all agents
        root/agents/<id>/meta.json       name, owner, schema, access policy
        root/agents/<id>/refs/<branch>   version id the branch points to
        root/agents/<id>/audit.jsonl     append-only access log
        root/agents/<id>/.lock           serializes ref + meta updates
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "agents").mkdir(parents=True, exist_ok=True)

    # ── objects ───────────────────────────────────────────────────────────

    def _obj_path(self, vid: str) -> Path:
        return self.root / "objects" / vid[:2] / f"{vid[2:]}.json"

    def put(self, version: Version) -> None:
        p = self._obj_path(version.id)
        if not p.exists():  # immutable: identical content always has the same id
            atomic_write(p, canonical_json(version.content()))

    def get(self, vid: str, verify: bool = True) -> Version:
        p = self._obj_path(vid)
        try:
            content = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise KeyError(f"No version {vid}") from None
        if verify and content_id(content) != vid:
            raise IntegrityError(f"Version {vid[:12]} was modified on disk (hash mismatch).")
        return Version.from_content(vid, content)

    def resolve_prefix(self, prefix: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{4,64}", prefix):
            raise KeyError(f"'{prefix}' is neither a branch nor a version id.")
        if len(prefix) == 64:
            return prefix
        matches = [f"{d.name}{f.stem}" for d in (self.root / "objects").glob(prefix[:2])
                   for f in d.glob(f"{prefix[2:]}*.json")]
        if len(matches) != 1:
            raise KeyError(f"Version prefix '{prefix}' matches {len(matches)} versions.")
        return matches[0]

    # ── agents, refs, meta ────────────────────────────────────────────────

    def agent_dir(self, agent_id: str) -> Path:
        return self.root / "agents" / agent_id

    def lock(self, agent_id: str) -> FileLock:
        return FileLock(self.agent_dir(agent_id) / ".lock")

    def read_meta(self, agent_id: str) -> dict[str, Any]:
        p = self.agent_dir(agent_id) / "meta.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise KeyError(f"No agent '{agent_id}'") from None

    def write_meta(self, agent_id: str, meta: dict[str, Any]) -> None:
        atomic_write(self.agent_dir(agent_id) / "meta.json", json.dumps(meta, indent=2))

    def list_agent_ids(self) -> list[str]:
        return sorted(p.name for p in (self.root / "agents").iterdir()
                      if (p / "meta.json").exists())

    def read_ref(self, agent_id: str, branch: str) -> str | None:
        if not _BRANCH_RE.match(branch):
            return None  # never let a ref name escape the refs directory
        p = self.agent_dir(agent_id) / "refs" / branch
        try:
            return p.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def list_refs(self, agent_id: str) -> dict[str, str]:
        d = self.agent_dir(agent_id) / "refs"
        return {p.name: p.read_text().strip() for p in sorted(d.glob("*"))} if d.exists() else {}

    def cas_ref(self, agent_id: str, branch: str, expected: str | None, new: str) -> None:
        """Move a branch to `new` only if it still points at `expected`. Caller holds the lock."""
        if not _BRANCH_RE.match(branch):
            raise ValueError(f"Invalid branch name '{branch}' (use letters, digits, . _ -).")
        actual = self.read_ref(agent_id, branch)
        if actual != expected:
            raise ConflictError(branch, expected, actual)
        atomic_write(self.agent_dir(agent_id) / "refs" / branch, new + "\n")

    def append_audit(self, agent_id: str, entry: dict[str, Any]) -> None:
        p = self.agent_dir(agent_id) / "audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read_audit(self, agent_id: str) -> list[dict[str, Any]]:
        p = self.agent_dir(agent_id) / "audit.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]

    # ── history ───────────────────────────────────────────────────────────

    def ancestors(self, vid: str) -> dict[str, int]:
        """All ancestors of a version (inclusive) with their distance."""
        seen: dict[str, int] = {}
        frontier = [(vid, 0)]
        while frontier:
            v, d = frontier.pop()
            if v in seen and seen[v] <= d:
                continue
            seen[v] = d
            frontier.extend((p, d + 1) for p in self.get(v, verify=False).parents)
        return seen

    def merge_base(self, a: str, b: str) -> str | None:
        """Nearest common ancestor of two versions."""
        anc_a = self.ancestors(a)
        anc_b = self.ancestors(b)
        common = set(anc_a) & set(anc_b)
        if not common:
            return None
        return min(common, key=lambda v: (anc_a[v] + anc_b[v], v))
