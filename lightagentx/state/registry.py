"""AgentRegistry — persistent, addressable, versioned and permissioned agents."""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..agents.single import SingleAgent
from ..hooks import HookRegistry
from ..llm.base import BaseLLM
from ..memory.buffer import BufferMemory
from ..tools.base import BaseTool, tool
from .access import AccessDenied, AccessPolicy
from .merge import MergeConflictError, Resolver, merge_state, merge_transcripts
from .store import ConflictError, IntegrityError, Version, VersionStore, current_provenance

_TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}
_MAX_MERGE_RETRIES = 20
_OPTIMISTIC_TRIES = 3


def _validate(schema: dict[str, str], state: dict[str, Any]) -> None:
    for key, type_name in schema.items():
        if key in state and state[key] is not None and not isinstance(state[key], _TYPES[type_name]):
            raise TypeError(f"State key '{key}' must be {type_name}, got {type(state[key]).__name__}")


def _memory_messages(agent: SingleAgent) -> list[dict[str, Any]]:
    return [m for m in agent.memory.get_messages() if m.get("role") != "system"]


def _load_messages(memory: BufferMemory, messages: list[dict[str, Any]]) -> None:
    for m in messages:
        kwargs: dict[str, Any] = {}
        if "tool_call_id" in m:
            kwargs["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            kwargs["tool_calls"] = [
                {"id": tc["id"], "name": tc["function"]["name"],
                 "arguments": tc["function"]["arguments"]}
                for tc in m["tool_calls"]
            ]
        memory.add_message(m["role"], m.get("content"), **kwargs)  # keep None as None


class Session:
    """
    A live attachment of one process to one agent branch.

    Use as a context manager: changes are committed on a clean exit and
    discarded if the block raises.
    """

    def __init__(self, registry: "AgentRegistry", agent_id: str, principal: str, branch: str,
                 base: Version, llm: BaseLLM | None, tools: list[BaseTool], *,
                 read_only: bool, on_conflict: str, merge_strategy: str | Resolver,
                 transcript_strategy: str, summarizer: Callable[[list[dict]], str] | None,
                 workflow: str, state_tools: bool, hooks: HookRegistry | None,
                 verbose: bool, sandbox: Any = None):
        self.registry = registry
        self.agent_id = agent_id
        self.principal = principal
        self.branch = branch
        self.read_only = read_only
        self.on_conflict = on_conflict
        self.merge_strategy = merge_strategy
        self.transcript_strategy = transcript_strategy
        self.summarizer = summarizer
        self.workflow = workflow
        self.llm = llm
        self.tools = list(tools)
        self.hooks = hooks
        self.verbose = verbose
        self.state_tools = state_tools
        self.sandbox = sandbox
        self.committed: list[Version] = []
        self._load(base)

    # ── loading ───────────────────────────────────────────────────────────

    def _load(self, base: Version) -> None:
        policy = self.registry._policy(self.agent_id)
        self.base = base
        self.redacted = not policy.can(self.principal, "private")
        self._loaded_transcript = policy.view_transcript(self.principal, list(base.transcript))
        self._loaded_state = policy.view_state(self.principal, base.state)
        self.state: dict[str, Any] = copy.deepcopy(self._loaded_state)
        self.agent: SingleAgent | None = None
        if self.llm is not None:
            memory = BufferMemory(max_messages=10**9)
            extra = self._make_state_tools() if self.state_tools else []
            self.agent = SingleAgent(
                name=base.config.get("name", self.agent_id), llm=self.llm,
                tools=self.tools + extra, memory=memory,
                system_prompt=self._system_prompt(base),
                description=base.config.get("description", ""),
                max_iterations=base.config.get("max_iterations", 10),
                verbose=self.verbose, hooks=self.hooks, sandbox=self.sandbox,
            )
            _load_messages(memory, self._loaded_transcript)

    def _system_prompt(self, base: Version) -> str:
        prompt = base.config.get("system_prompt", "You are a helpful AI assistant.")
        notes = self.state.get("_merge_notes") or []
        digests = [n for n in notes if n.get("digest")]
        if digests:
            prompt += "\n\nContext merged from parallel sessions:\n" + "\n".join(
                f"- ({n['from']}) {n['digest']}" for n in digests[-5:])
        return prompt

    # ── using the agent ───────────────────────────────────────────────────

    def run(self, text: str) -> str:
        if self.agent is None:
            raise RuntimeError("Session was attached without an LLM; it can only read and edit state.")
        return self.agent.run(text)

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return _memory_messages(self.agent) if self.agent else list(self._loaded_transcript)

    def _make_state_tools(self) -> list[BaseTool]:
        policy = self.registry._policy(self.agent_id)
        schema = self.registry._meta(self.agent_id).get("schema", {})
        session = self

        @tool(risk="low")  # touches only this agent's stored state, never the OS
        def get_state(key: str = "") -> str:
            """Read the agent's persistent structured state (all keys, or one key).

            Args:
                key: State key to read; empty for everything.
            """
            data = session.state if not key else {key: session.state.get(key)}
            return json.dumps(data, default=str)

        @tool(risk="low")  # touches only this agent's stored state, never the OS
        def set_state(key: str, value_json: str) -> str:
            """Set a key in the agent's persistent structured state. It is saved when the session commits.

            Args:
                key: State key.
                value_json: New value as JSON (e.g. "\\"ABC\\"", "[1, 2]", "{\\"a\\": 1}").
            """
            if session.redacted and key in policy.private_keys:
                raise AccessDenied(f"'{key}' is a private state key.")
            value = json.loads(value_json)
            _validate(schema, {key: value})
            session.state[key] = value
            return f"state[{key!r}] updated"

        @tool(risk="low")  # touches only this agent's stored state, never the OS
        def append_state(key: str, item_json: str) -> str:
            """Append an item to a list in the agent's persistent structured state.

            Args:
                key: State key holding a list.
                item_json: Item to append, as JSON.
            """
            if session.redacted and key in policy.private_keys:
                raise AccessDenied(f"'{key}' is a private state key.")
            session.state.setdefault(key, []).append(json.loads(item_json))
            _validate(schema, {key: session.state[key]})
            return f"appended to state[{key!r}]"

        return [get_state, set_state, append_state]

    # ── committing ────────────────────────────────────────────────────────

    def _changes(self) -> tuple[list[dict], dict[str, Any]] | None:
        """Write the session's delta back onto the FULL (unredacted) base state."""
        current = self.transcript
        loaded = self._loaded_transcript
        if current[:len(loaded)] != loaded:
            raise RuntimeError("Session transcript no longer extends the loaded one; cannot commit.")
        suffix = current[len(loaded):]

        policy = self.registry._policy(self.agent_id)
        new_state = copy.deepcopy(self.base.state)
        changed = False
        for k in set(self._loaded_state) | set(self.state):
            if self.state.get(k, ...) != self._loaded_state.get(k, ...):
                if self.redacted and k in policy.private_keys:
                    raise AccessDenied(f"'{k}' is a private state key.")
                changed = True
                if k in self.state:
                    new_state[k] = copy.deepcopy(self.state[k])
                else:
                    new_state.pop(k, None)
        if not suffix and not changed:
            return None
        return list(self.base.transcript) + suffix, new_state

    def commit(self, message: str = "") -> Version | None:
        """Save this session's changes as a new version. Returns None if nothing changed."""
        if self.read_only:
            raise AccessDenied("This session is read-only.")
        changes = self._changes()
        if changes is None:
            return None
        transcript, state = changes
        version = self.registry._commit(self, transcript, state, message)
        self.committed.append(version)
        self._load(version)  # continue from what was actually stored
        return version

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self.read_only:
            self.commit("session end")
        elif exc_type is not None:
            self.registry._audit(self.agent_id, self.principal, "session_aborted", self.branch,
                                 self.base.id, "discarded", repr(exc)[:200])


class AgentRegistry:
    """
    A shared home for persistent agents. Any process pointed at the same
    `root` directory can discover agents, attach to them, and continue
    from their exact state.

    Usage::

        reg = AgentRegistry("~/.lightx/agents")
        aid = reg.create("SupportBot", owner="alice", system_prompt="...",
                         schema={"customer": "str", "decisions": "list"})

        # any process, any time later:
        with reg.attach(aid, principal="alice", llm=llm) as s:
            s.run("Customer ABC wants a refund")
            s.state["customer"] = "ABC"
        # -> committed as a new immutable version with provenance
    """

    def __init__(self, root: str | Path = "~/.lightx/agents"):
        self.store = VersionStore(root)

    # ── internals ─────────────────────────────────────────────────────────

    def _meta(self, agent_id: str) -> dict[str, Any]:
        return self.store.read_meta(agent_id)

    def _policy(self, agent_id: str) -> AccessPolicy:
        return AccessPolicy.from_dict(self._meta(agent_id)["policy"])

    def _audit(self, agent_id: str, principal: str, action: str, branch: str | None,
               version: str | None, decision: str, detail: str = "") -> None:
        self.store.append_audit(agent_id, {
            "ts": time.time(), "principal": principal, "action": action, "branch": branch,
            "version": version, "decision": decision, "detail": detail,
        })

    def _require(self, agent_id: str, principal: str, right: str, action: str,
                 branch: str | None = None) -> AccessPolicy:
        policy = self._policy(agent_id)
        if not policy.can(principal, right):
            self._audit(agent_id, principal, action, branch, None, "denied", f"needs '{right}'")
            raise AccessDenied(f"'{principal}' lacks '{right}' on agent {agent_id}.")
        return policy

    def _resolve(self, agent_id: str, ref: str) -> Version:
        head = self.store.read_ref(agent_id, ref)
        vid = head or self.store.resolve_prefix(ref)
        version = self.store.get(vid)
        if version.agent_id != agent_id:
            raise KeyError(f"Version {ref} does not belong to agent {agent_id}.")
        return version

    def _new_version(self, agent_id: str, parents: tuple[str, ...], principal: str,
                     workflow: str, message: str, config: dict, transcript: list,
                     state: dict, tool_manifest: tuple[str, ...]) -> Version:
        return Version.create(
            agent_id=agent_id, parents=parents, created_at=time.time(), message=message,
            provenance=current_provenance(principal, workflow), config=config,
            transcript=transcript, state=state, tool_manifest=tool_manifest,
        )

    # ── creating & discovering ────────────────────────────────────────────

    def create(self, name: str, owner: str, system_prompt: str = "You are a helpful AI assistant.",
               description: str = "", schema: dict[str, str] | None = None,
               policy: AccessPolicy | None = None, state: dict[str, Any] | None = None,
               transcript: list[dict] | None = None, max_iterations: int = 10,
               agent_id: str | None = None) -> str:
        """Register a new persistent agent. Returns its agent id."""
        schema = schema or {}
        for t in schema.values():
            if t not in _TYPES:
                raise ValueError(f"Unknown schema type '{t}'. Use one of {sorted(_TYPES)}.")
        state = state or {}
        _validate(schema, state)
        agent_id = agent_id or f"agt_{uuid.uuid4().hex[:12]}"
        policy = policy or AccessPolicy(owner=owner)
        if policy.owner != owner:
            raise ValueError("policy.owner must match owner")
        config = {"name": name, "description": description, "system_prompt": system_prompt,
                  "max_iterations": max_iterations}
        root = self._new_version(agent_id, (), owner, "create", "created", config,
                                 transcript or [], state, ())
        with self.store.lock(agent_id):
            if (self.store.agent_dir(agent_id) / "meta.json").exists():
                raise ValueError(f"Agent {agent_id} already exists.")
            self.store.write_meta(agent_id, {
                "agent_id": agent_id, "name": name, "description": description,
                "created_at": root.created_at, "owner": owner, "schema": schema,
                "policy": policy.to_dict(),
            })
            self.store.put(root)
            self.store.cas_ref(agent_id, "main", None, root.id)
        self._audit(agent_id, owner, "create", "main", root.id, "allowed")
        return agent_id

    def register(self, agent: SingleAgent, owner: str, **kwargs: Any) -> str:
        """Adopt an existing in-memory SingleAgent (config + conversation so far)."""
        return self.create(
            name=agent.name, owner=owner, system_prompt=agent.system_prompt,
            description=agent.description, max_iterations=agent.max_iterations,
            transcript=_memory_messages(agent), **kwargs,
        )

    def list_agents(self, principal: str | None = None) -> list[dict[str, Any]]:
        """Discover agents (only those `principal` may read, if given)."""
        out = []
        for aid in self.store.list_agent_ids():
            meta = self._meta(aid)
            if principal and not AccessPolicy.from_dict(meta["policy"]).can(principal, "read"):
                continue
            out.append({"agent_id": aid, "name": meta["name"], "description": meta["description"],
                        "owner": meta["owner"], "branches": self.store.list_refs(aid)})
        return out

    def find(self, name: str, principal: str | None = None) -> str:
        hits = [a["agent_id"] for a in self.list_agents(principal) if a["name"] == name]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} agents named '{name}'.")
        return hits[0]

    # ── attaching ─────────────────────────────────────────────────────────

    def attach(self, agent_id: str, principal: str, llm: BaseLLM | None = None,
               tools: list[BaseTool] | None = None, branch: str = "main", at: str | None = None,
               on_conflict: str = "merge", merge_strategy: str | Resolver = "raise",
               transcript_strategy: str = "append",
               summarizer: Callable[[list[dict]], str] | None = None,
               workflow: str = "", state_tools: bool = False,
               hooks: HookRegistry | None = None, verbose: bool = False,
               sandbox: Any = None) -> Session:
        """
        Attach this process to an agent and continue from its stored state.

        Args:
            branch: Branch to continue (and commit to).
            at: A specific version id/prefix to start from (time travel). Read-only
                unless you fork it into a branch first.
            on_conflict: When the branch moved meanwhile — "merge" (three-way),
                "fork" (commit to a new branch) or "reject" (raise ConflictError).
            merge_strategy: How to settle a state key both sides changed:
                "raise", "ours", "theirs" or callable(path, base, ours, theirs).
            transcript_strategy: "append" or "digest" (needs `summarizer`).
            state_tools: Give the LLM get_state/set_state/append_state tools.
            sandbox: Gate every tool call of this session through a Sandbox.
        """
        if on_conflict not in ("merge", "fork", "reject"):
            raise ValueError("on_conflict must be 'merge', 'fork' or 'reject'")
        right = "run" if llm is not None else "read"
        self._require(agent_id, principal, right, "attach", branch)
        base = self._resolve(agent_id, at or branch)
        read_only = at is not None or not self._policy(agent_id).can(principal, "write")
        if read_only and self._policy(agent_id).can(principal, "fork") and at is None:
            read_only = False  # commits will go to a personal fork branch
        tools = tools or []
        missing = set(base.tool_manifest) - {t.name for t in tools}
        if missing and llm is not None:
            import warnings
            warnings.warn(f"Agent expects tools {sorted(missing)} that were not provided.",
                          UserWarning, stacklevel=2)
        self._audit(agent_id, principal, "attach", branch, base.id, "allowed",
                    f"redacted={not self._policy(agent_id).can(principal, 'private')}")
        return Session(self, agent_id, principal, branch, base, llm, tools,
                       read_only=read_only, on_conflict=on_conflict,
                       merge_strategy=merge_strategy, transcript_strategy=transcript_strategy,
                       summarizer=summarizer, workflow=workflow, state_tools=state_tools,
                       hooks=hooks, verbose=verbose, sandbox=sandbox)

    def _commit(self, s: Session, transcript: list[dict], state: dict[str, Any],
                message: str) -> Version:
        aid = s.agent_id
        policy = self._policy(aid)
        _validate(self._meta(aid).get("schema", {}), state)
        branch = s.branch
        if not policy.can_write_branch(s.principal, branch):
            if not policy.can(s.principal, "fork"):
                self._audit(aid, s.principal, "commit", branch, None, "denied", "no write/fork")
                raise AccessDenied(f"'{s.principal}' may not commit to '{branch}'.")
            branch = f"{s.principal.replace('/', '-')}.{branch}"  # personal fork branch
        tool_manifest = tuple(sorted(t.name for t in s.tools))
        ours = self._new_version(aid, (s.base.id,), s.principal, s.workflow, message or "commit",
                                 s.base.config, transcript, state, tool_manifest)
        self.store.put(ours)

        # A few lock-free attempts first (cheap when uncontended); after that,
        # read-merge-swap under the agent lock so the commit is guaranteed to land.
        for attempt in range(_MAX_MERGE_RETRIES):
            if attempt < _OPTIMISTIC_TRIES:
                result = self._try_commit(s, ours, branch, message, locked=False)
            else:
                with self.store.lock(aid):
                    result = self._try_commit(s, ours, branch, message, locked=True)
            if result is not None:
                return result
        raise ConflictError(branch, s.base.id, self.store.read_ref(aid, branch))

    def _try_commit(self, s: Session, ours: Version, branch: str, message: str,
                    locked: bool) -> Version | None:
        """One attempt to land `ours` on `branch`. Returns None if the branch moved under us."""
        aid = s.agent_id

        def swap(name: str, expected: str | None, new: str) -> bool:
            try:
                if locked:
                    self.store.cas_ref(aid, name, expected, new)
                else:
                    with self.store.lock(aid):
                        self.store.cas_ref(aid, name, expected, new)
                return True
            except ConflictError:
                return False

        head = self.store.read_ref(aid, branch)
        expected = s.base.id if branch == s.branch else head
        if head == expected or head is None:
            if not swap(branch, head, ours.id):
                return None
            self._audit(aid, s.principal, "commit", branch, ours.id, "allowed", message)
            s.branch = branch  # stays on a personal fork branch if one was used
            return ours

        # The branch moved since this session attached.
        if s.on_conflict == "reject":
            self._audit(aid, s.principal, "commit", branch, ours.id, "conflict", "rejected")
            raise ConflictError(branch, s.base.id, head)
        if s.on_conflict == "fork":
            fork = f"{branch}.fork-{ours.id[:8]}"
            swap(fork, None, ours.id)
            self._audit(aid, s.principal, "commit", fork, ours.id, "forked", f"{branch} had moved")
            s.branch = fork
            return ours
        merged = self._merge_versions(aid, s.principal, s.workflow, theirs=self.store.get(head),
                                      ours=ours, label=f"{branch}@{head[:8]}",
                                      strategy=s.merge_strategy,
                                      transcript_strategy=s.transcript_strategy,
                                      summarizer=s.summarizer)
        if not swap(branch, head, merged.id):
            return None  # someone committed again while we merged; retry on the new head
        self._audit(aid, s.principal, "commit", branch, merged.id, "merged",
                    f"auto-merged {ours.id[:8]} into {head[:8]}")
        s.branch = branch
        return merged

    def _merge_versions(self, aid: str, principal: str, workflow: str, theirs: Version,
                        ours: Version, label: str, strategy: str | Resolver,
                        transcript_strategy: str,
                        summarizer: Callable[[list[dict]], str] | None,
                        first_parent: Version | None = None) -> Version:
        base_id = self.store.merge_base(theirs.id, ours.id)
        base_state = self.store.get(base_id).state if base_id else {}
        state, resolved = merge_state(base_state, ours.state, theirs.state, strategy)
        transcript, note = merge_transcripts(list(ours.transcript), list(theirs.transcript),
                                             label, transcript_strategy, summarizer)
        if note or resolved:
            entry = dict(note)
            if resolved:
                entry["resolved_conflicts"] = [str(c) for c in resolved]
            state["_merge_notes"] = list(state.get("_merge_notes", [])) + [entry]
        first = first_parent or theirs
        second = ours if first is theirs else theirs
        merged = self._new_version(
            aid, (first.id, second.id), principal, workflow, f"merge {ours.id[:8]} into {label}",
            theirs.config, transcript, state,
            tuple(sorted(set(theirs.tool_manifest) | set(ours.tool_manifest))))
        self.store.put(merged)
        return merged

    # ── branches: fork & merge ────────────────────────────────────────────

    def fork(self, agent_id: str, principal: str, new_branch: str, from_ref: str = "main") -> str:
        """Create a branch at `from_ref` (a branch or a version id)."""
        self._require(agent_id, principal, "fork", "fork", new_branch)
        src = self._resolve(agent_id, from_ref)
        with self.store.lock(agent_id):
            self.store.cas_ref(agent_id, new_branch, None, src.id)
        self._audit(agent_id, principal, "fork", new_branch, src.id, "allowed", f"from {from_ref}")
        return new_branch

    def merge(self, agent_id: str, principal: str, source: str, into: str = "main",
              strategy: str | Resolver = "raise", transcript_strategy: str = "append",
              summarizer: Callable[[list[dict]], str] | None = None) -> Version:
        """Merge branch `source` into branch `into` (fast-forward when possible)."""
        policy = self._require(agent_id, principal, "merge", "merge", into)
        if not policy.can_write_branch(principal, into):
            self._audit(agent_id, principal, "merge", into, None, "denied", "cannot write target")
            raise AccessDenied(f"'{principal}' may not write to '{into}'.")
        for attempt in range(_MAX_MERGE_RETRIES):
            locked = attempt >= _OPTIMISTIC_TRIES
            if locked:
                with self.store.lock(agent_id):
                    result = self._try_merge(agent_id, principal, source, into, strategy,
                                             transcript_strategy, summarizer, locked=True)
            else:
                result = self._try_merge(agent_id, principal, source, into, strategy,
                                         transcript_strategy, summarizer, locked=False)
            if result is not None:
                return result
        raise ConflictError(into, None, self.store.read_ref(agent_id, into))

    def _try_merge(self, agent_id: str, principal: str, source: str, into: str,
                   strategy: str | Resolver, transcript_strategy: str,
                   summarizer: Callable[[list[dict]], str] | None, locked: bool) -> Version | None:
        target = self._resolve(agent_id, into)
        src = self._resolve(agent_id, source)
        if src.id in self.store.ancestors(target.id):
            return target  # already merged
        if target.id in self.store.ancestors(src.id):
            result = src  # fast-forward
        else:
            result = self._merge_versions(agent_id, principal, "merge", theirs=src, ours=target,
                                          label=source, strategy=strategy,
                                          transcript_strategy=transcript_strategy,
                                          summarizer=summarizer, first_parent=target)
        try:
            if locked:
                self.store.cas_ref(agent_id, into, target.id, result.id)
            else:
                with self.store.lock(agent_id):
                    self.store.cas_ref(agent_id, into, target.id, result.id)
        except ConflictError:
            return None
        self._audit(agent_id, principal, "merge", into, result.id, "allowed", f"from {source}")
        return result

    # ── inspection ────────────────────────────────────────────────────────

    def branches(self, agent_id: str, principal: str) -> dict[str, str]:
        self._require(agent_id, principal, "read", "branches")
        return self.store.list_refs(agent_id)

    def read(self, agent_id: str, principal: str, ref: str = "main") -> dict[str, Any]:
        """The state and transcript at a branch/version, as `principal` may see it."""
        policy = self._require(agent_id, principal, "read", "read", ref)
        v = self._resolve(agent_id, ref)
        self._audit(agent_id, principal, "read", ref, v.id, "allowed")
        return {"version": v.id, "state": policy.view_state(principal, v.state),
                "transcript": policy.view_transcript(principal, list(v.transcript)),
                "config": v.config, "provenance": v.provenance, "parents": list(v.parents)}

    def log(self, agent_id: str, principal: str, ref: str = "main", limit: int = 50) -> list[Version]:
        """History reachable from `ref`, newest first."""
        self._require(agent_id, principal, "read", "log", ref)
        head = self._resolve(agent_id, ref)
        versions = [self.store.get(v) for v in self.store.ancestors(head.id)]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)[:limit]

    def diff(self, agent_id: str, principal: str, a: str, b: str) -> dict[str, Any]:
        policy = self._require(agent_id, principal, "read", "diff")
        va, vb = self._resolve(agent_id, a), self._resolve(agent_id, b)
        sa, sb = policy.view_state(principal, va.state), policy.view_state(principal, vb.state)
        return {
            "state": {k: (sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb))
                      if sa.get(k) != sb.get(k)},
            "transcript_messages": (len(va.transcript), len(vb.transcript)),
        }

    def verify(self, agent_id: str) -> int:
        """Re-hash every version reachable from every branch. Returns how many were checked."""
        seen: set[str] = set()
        for head in self.store.list_refs(agent_id).values():
            stack = [head]
            while stack:
                vid = stack.pop()
                if vid in seen:
                    continue
                v = self.store.get(vid, verify=True)  # raises IntegrityError on tampering
                seen.add(vid)
                stack.extend(v.parents)
        return len(seen)

    def audit_log(self, agent_id: str, principal: str) -> list[dict[str, Any]]:
        self._require(agent_id, principal, "admin", "audit")
        return self.store.read_audit(agent_id)

    # ── administration ────────────────────────────────────────────────────

    def grant(self, agent_id: str, by: str, principal: str, rights: set[str] | list[str],
              write_branches: list[str] | None = None) -> None:
        self._require(agent_id, by, "admin", "grant")
        with self.store.lock(agent_id):
            meta = self._meta(agent_id)
            policy = AccessPolicy.from_dict(meta["policy"])
            policy.grants[principal] = set(policy.grants.get(principal, set())) | set(rights)
            if write_branches is not None:
                policy.write_branches[principal] = list(write_branches)
            meta["policy"] = policy.to_dict()
            self.store.write_meta(agent_id, meta)
        self._audit(agent_id, by, "grant", None, None, "allowed", f"{principal}: {sorted(rights)}")

    def revoke(self, agent_id: str, by: str, principal: str) -> None:
        self._require(agent_id, by, "admin", "revoke")
        with self.store.lock(agent_id):
            meta = self._meta(agent_id)
            policy = AccessPolicy.from_dict(meta["policy"])
            policy.grants.pop(principal, None)
            policy.write_branches.pop(principal, None)
            meta["policy"] = policy.to_dict()
            self.store.write_meta(agent_id, meta)
        self._audit(agent_id, by, "revoke", None, None, "allowed", principal)

    def set_private_keys(self, agent_id: str, by: str, keys: set[str] | list[str]) -> None:
        self._require(agent_id, by, "admin", "set_private_keys")
        with self.store.lock(agent_id):
            meta = self._meta(agent_id)
            policy = AccessPolicy.from_dict(meta["policy"])
            policy.private_keys = set(keys)
            meta["policy"] = policy.to_dict()
            self.store.write_meta(agent_id, meta)
        self._audit(agent_id, by, "set_private_keys", None, None, "allowed", str(sorted(keys)))
