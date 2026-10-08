"""Agent archives — export an agent (history, config, tools) to one file and import it elsewhere.

Archive layout (a gzip tar, conventionally ``<name>.lxagent``)::

    manifest.json        format, source, branches, SHA-256 of every other file
    meta.json            name, owner, schema, access policy
    objects/<id>.json    versions, byte-for-byte as stored (content-addressed)
    tools/<file>.py      the tools' source code

Importing data never executes anything. Tool files are code written by
whoever made the archive, so loading them is a separate, explicit step
(`AgentRegistry.load_tools(..., trust=True)`).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tarfile
import tempfile
import time
import warnings
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable

from .access import AccessPolicy
from .store import IntegrityError, Version, canonical_json, content_id, current_provenance

if TYPE_CHECKING:
    from ..tools.base import BaseTool
    from .registry import AgentRegistry

ARCHIVE_FORMAT = "lightx-agent"
ARCHIVE_VERSION = 1
_MAX_MEMBER_BYTES = 256 * 1024 * 1024


class ArchiveError(ValueError):
    """The file is not a valid agent archive (wrong format, unsafe paths, missing files)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── tools: find source files and the tool names they define (without running them) ──

def _tool_source(item: "BaseTool | str | Path") -> Path:
    if isinstance(item, (str, Path)):
        path = Path(item).expanduser().resolve()
    else:
        src = inspect.getsourcefile(item.func)
        if src is None:
            raise ArchiveError(f"Can't find the source file of tool '{item.name}'.")
        path = Path(src).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ArchiveError(f"Tool source must be an existing .py file: {path}")
    return path


def declared_tool_names(source: str) -> set[str]:
    """Names of module-level functions decorated with @tool / @tool(...). Static: no execution."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if (isinstance(target, ast.Name) and target.id == "tool") or (
                        isinstance(target, ast.Attribute) and target.attr == "tool"):
                    names.add(node.name)
    return names


# ── export ───────────────────────────────────────────────────────────────

def export_agent(
    registry: "AgentRegistry",
    agent_id: str,
    principal: str,
    path: str | Path,
    *,
    branches: Iterable[str] | None = None,
    tools: Iterable["BaseTool | str | Path"] = (),
    redact: bool = False,
) -> Path:
    from .. import __version__

    store = registry.store
    if redact:
        policy = registry._require(agent_id, principal, "read", "export", None)
    else:
        policy = registry._require(agent_id, principal, "admin", "export", None)
    meta = registry._meta(agent_id)
    refs = store.list_refs(agent_id)
    wanted = list(branches) if branches is not None else list(refs)
    missing = [b for b in wanted if b not in refs]
    if missing:
        raise KeyError(f"No such branch(es): {missing}")

    files: dict[str, bytes] = {}
    out_refs: dict[str, str] = {}

    if not redact:
        # Full history: every version reachable from the exported branches, unchanged.
        for branch in wanted:
            out_refs[branch] = refs[branch]
            for vid in store.ancestors(refs[branch]):
                if f"objects/{vid}.json" not in files:
                    files[f"objects/{vid}.json"] = canonical_json(store.get(vid).content()).encode()
    else:
        # Redacted: one new root version per branch, built from what `principal` may see.
        for branch in wanted:
            head = store.get(refs[branch])
            v = Version.create(
                agent_id=agent_id, parents=(), created_at=head.created_at,
                message=f"redacted export of '{branch}' (history not included)",
                provenance=current_provenance(principal, "export"), config=head.config,
                transcript=policy.view_transcript(principal, list(head.transcript)),
                state=policy.view_state(principal, head.state),
                tool_manifest=head.tool_manifest,
            )
            out_refs[branch] = v.id
            files[f"objects/{v.id}.json"] = canonical_json(v.content()).encode()

    # Tool sources, plus a static check that they define what the agent expects.
    sources: dict[str, Path] = {}
    for item in tools:
        src = _tool_source(item)
        existing = sources.get(src.name)
        if existing and existing != src:
            raise ArchiveError(f"Two different tool files are both named '{src.name}'.")
        sources[src.name] = src
    defined: set[str] = set()
    for name, src in sources.items():
        data = src.read_bytes()
        files[f"tools/{name}"] = data
        defined |= declared_tool_names(data.decode("utf-8"))
    expected = {t for b in wanted for t in store.get(refs[b]).tool_manifest}
    if sources and expected - defined:
        warnings.warn(f"Exported tool files don't define {sorted(expected - defined)}, "
                      f"which the agent has used.", UserWarning, stacklevel=3)

    meta_out = dict(meta)
    files["meta.json"] = json.dumps(meta_out, indent=2).encode()
    manifest = {
        "format": ARCHIVE_FORMAT,
        "format_version": ARCHIVE_VERSION,
        "agent_id": agent_id,
        "name": meta["name"],
        "exported_at": time.time(),
        "exported_by": current_provenance(principal, "export"),
        "lightagentx_version": __version__,
        "history": "squashed" if redact else "full",
        "redacted": redact,
        "branches": out_refs,
        "files": {name: _sha256(data) for name, data in sorted(files.items())},
    }

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".lxagent")
    try:
        with open(fd, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
            for name, data in [("manifest.json", json.dumps(manifest, indent=2).encode()),
                               *sorted(files.items())]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = int(manifest["exported_at"])
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    registry._audit(agent_id, principal, "export", None, None, "allowed",
                    f"{path.name}: {len(out_refs)} branch(es), "
                    f"{sum(n.startswith('objects/') for n in files)} version(s), "
                    f"{len(sources)} tool file(s), redacted={redact}")
    return path


# ── import ───────────────────────────────────────────────────────────────

def _read_archive(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read every member into memory after validating names. Nothing touches disk."""
    try:
        tar = tarfile.open(path, mode="r:gz")
    except (tarfile.TarError, OSError) as e:
        raise ArchiveError(f"Not a readable agent archive: {e}") from None
    members: dict[str, bytes] = {}
    with tar:
        for m in tar.getmembers():
            p = PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or ".." in p.parts or len(p.parts) > 2:
                raise ArchiveError(f"Unsafe or unexpected archive entry: {m.name!r}")
            if m.size > _MAX_MEMBER_BYTES:
                raise ArchiveError(f"Archive entry too large: {m.name}")
            if m.name in members:
                raise ArchiveError(f"Duplicate archive entry: {m.name}")
            members[m.name] = tar.extractfile(m).read()
    if "manifest.json" not in members:
        raise ArchiveError("Archive has no manifest.json")
    manifest = json.loads(members.pop("manifest.json"))
    if manifest.get("format") != ARCHIVE_FORMAT:
        raise ArchiveError("Not a LightAgentX agent archive.")
    if manifest.get("format_version") != ARCHIVE_VERSION:
        raise ArchiveError(f"Unsupported archive version {manifest.get('format_version')}.")

    listed = manifest.get("files", {})
    if set(listed) != set(members):
        raise IntegrityError(f"Archive contents don't match its manifest "
                             f"(extra: {sorted(set(members) - set(listed))}, "
                             f"missing: {sorted(set(listed) - set(members))}).")
    for name, data in members.items():
        if _sha256(data) != listed[name]:
            raise IntegrityError(f"Archive file '{name}' was modified (hash mismatch).")
    return manifest, members


def _rekey(versions: dict[str, dict[str, Any]], new_agent_id: str) -> dict[str, str]:
    """Rewrite versions for a new agent id (ids change, so parents are remapped). old->new."""
    mapping: dict[str, str] = {}
    remaining = dict(versions)
    while remaining:
        ready = [vid for vid, c in remaining.items() if all(p in mapping for p in c["parents"])]
        if not ready:
            raise IntegrityError("Version history has missing parents or a cycle.")
        for vid in ready:
            c = dict(remaining.pop(vid))
            c["agent_id"] = new_agent_id
            c["parents"] = [mapping[p] for p in c["parents"]]
            mapping[vid] = content_id(c)
            versions[vid] = c
    return mapping


def import_agent(
    registry: "AgentRegistry",
    path: str | Path,
    principal: str,
    *,
    agent_id: str | None = None,
    owner: str | None = None,
) -> str:
    store = registry.store
    path = Path(path).expanduser()
    manifest, members = _read_archive(path)
    source_id = manifest["agent_id"]

    # Every version must hash to its own id and belong to the archived agent.
    versions: dict[str, dict[str, Any]] = {}
    for name, data in members.items():
        if name.startswith("objects/"):
            vid = PurePosixPath(name).stem
            content = json.loads(data)
            if content_id(content) != vid:
                raise IntegrityError(f"Version {vid[:12]} doesn't match its hash.")
            if content.get("agent_id") != source_id:
                raise IntegrityError(f"Version {vid[:12]} belongs to another agent.")
            versions[vid] = content
    for vid, c in versions.items():
        absent = [p for p in c["parents"] if p not in versions]
        if absent:
            raise IntegrityError(f"Version {vid[:12]} references missing parent(s).")
    for branch, vid in manifest["branches"].items():
        if vid not in versions:
            raise IntegrityError(f"Branch '{branch}' points to a version not in the archive.")

    target_id = agent_id or source_id
    if (store.agent_dir(target_id) / "meta.json").exists():
        raise ValueError(f"Agent {target_id} already exists here. "
                         f"Pass agent_id='...' to import it as a copy.")

    mapping = {vid: vid for vid in versions}
    if target_id != source_id:
        mapping = _rekey(versions, target_id)

    meta = json.loads(members["meta.json"])
    meta["agent_id"] = target_id
    if owner:
        policy = AccessPolicy.from_dict(meta["policy"])
        policy.owner = owner
        meta["policy"] = policy.to_dict()
        meta["owner"] = owner

    with store.lock(target_id):
        if (store.agent_dir(target_id) / "meta.json").exists():
            raise ValueError(f"Agent {target_id} already exists here.")
        for old, content in versions.items():
            store.put(Version.from_content(mapping[old], content))
        store.write_meta(target_id, meta)
        for branch, vid in manifest["branches"].items():
            store.cas_ref(target_id, branch, None, mapping[vid])
        tools_dir = store.agent_dir(target_id) / "tools"
        for name, data in members.items():
            if name.startswith("tools/"):
                tools_dir.mkdir(parents=True, exist_ok=True)
                (tools_dir / PurePosixPath(name).name).write_bytes(data)
    registry._audit(target_id, principal, "import", None, None, "allowed",
                    f"from {path.name} (agent {source_id}, exported by "
                    f"{manifest['exported_by'].get('principal')}, history={manifest['history']})")
    return target_id


# ── loading tool code ────────────────────────────────────────────────────

def tool_files(registry: "AgentRegistry", agent_id: str) -> dict[str, str]:
    """Imported tool files and their SHA-256, for review before trusting them."""
    d = registry.store.agent_dir(agent_id) / "tools"
    return {p.name: _sha256(p.read_bytes()) for p in sorted(d.glob("*.py"))} if d.exists() else {}


def load_tools(registry: "AgentRegistry", agent_id: str, principal: str, *,
               trust: bool = False) -> list["BaseTool"]:
    from ..tools.base import BaseTool

    registry._require(agent_id, principal, "run", "load_tools", None)
    files = tool_files(registry, agent_id)
    if not files:
        return []
    if not trust:
        listing = "\n".join(f"  {n}  sha256={h[:16]}…" for n, h in files.items())
        raise PermissionError(
            "These tool files are Python code from the archive's author; loading them runs it.\n"
            f"{listing}\nReview them in {registry.store.agent_dir(agent_id) / 'tools'} "
            "and call load_tools(..., trust=True) to load.")

    found: dict[str, BaseTool] = {}
    for name in files:
        file = registry.store.agent_dir(agent_id) / "tools" / name
        module_name = f"lightx_agent_tools.{agent_id}.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        for value in vars(module).values():
            if isinstance(value, BaseTool):
                found.setdefault(value.name, value)
    registry._audit(agent_id, principal, "load_tools", None, None, "allowed",
                    f"{sorted(files)} -> {sorted(found)}")
    return [found[n] for n in sorted(found)]
