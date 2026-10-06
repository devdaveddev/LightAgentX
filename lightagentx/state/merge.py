"""Three-way merging of agent state: structured fields and conversation transcripts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

_MISSING = object()

# resolver(path, base, ours, theirs) -> merged value
Resolver = Callable[[str, Any, Any, Any], Any]


@dataclass
class FieldConflict:
    path: str
    base: Any
    ours: Any
    theirs: Any

    def __str__(self) -> str:
        show = lambda v: "<missing>" if v is _MISSING else repr(v)
        return f"{self.path}: base={show(self.base)} ours={show(self.ours)} theirs={show(self.theirs)}"


class MergeConflictError(RuntimeError):
    def __init__(self, conflicts: list[FieldConflict]):
        self.conflicts = conflicts
        super().__init__("Unresolved state conflicts:\n  " + "\n  ".join(map(str, conflicts)))


@dataclass
class MergeResult:
    state: dict[str, Any]
    transcript: list[dict[str, Any]]
    conflicts: list[FieldConflict] = field(default_factory=list)  # resolved ones, for the record


def _resolve(strategy: str | Resolver, c: FieldConflict) -> Any:
    if callable(strategy):
        return strategy(c.path, c.base, c.ours, c.theirs)
    if strategy == "ours":
        return c.ours
    if strategy == "theirs":
        return c.theirs
    raise MergeConflictError([c])


def merge_lists(base: list, ours: list, theirs: list) -> list:
    """
    Order-preserving three-way list merge.

    Items removed by either side are removed; items added by either side are
    kept (ours first, then theirs), without duplicates.
    """
    def key(x: Any) -> str:
        import json
        return json.dumps(x, sort_keys=True, default=str)

    base_k, ours_k, theirs_k = ({key(x) for x in lst} for lst in (base, ours, theirs))
    removed = {k for k in base_k if k not in ours_k or k not in theirs_k}
    out, seen = [], set()
    for item in list(ours) + list(theirs):
        k = key(item)
        if k in removed or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def merge_state(
    base: dict[str, Any],
    ours: dict[str, Any],
    theirs: dict[str, Any],
    strategy: str | Resolver = "raise",
    _path: str = "",
) -> tuple[dict[str, Any], list[FieldConflict]]:
    """
    Recursive three-way merge of structured state.

    - changed on one side only -> take that side
    - changed identically on both -> take it
    - dicts on all sides -> merge recursively
    - lists on all sides -> merge_lists (union of additions, removals respected)
    - otherwise -> conflict, settled by `strategy` ("raise", "ours", "theirs" or a callable)
    """
    merged: dict[str, Any] = {}
    resolved: list[FieldConflict] = []
    unresolved: list[FieldConflict] = []

    for k in sorted(set(base) | set(ours) | set(theirs), key=str):
        b, o, t = base.get(k, _MISSING), ours.get(k, _MISSING), theirs.get(k, _MISSING)
        path = f"{_path}.{k}" if _path else str(k)

        if o == t:
            value = o
        elif o == b:
            value = t
        elif t == b:
            value = o
        elif all(isinstance(x, dict) for x in (o, t)) and (b is _MISSING or isinstance(b, dict)):
            sub, sub_resolved = merge_state(
                {} if b is _MISSING else b, o, t, strategy, path)
            resolved.extend(sub_resolved)
            value = sub
        elif all(isinstance(x, list) for x in (o, t)) and (b is _MISSING or isinstance(b, list)):
            value = merge_lists([] if b is _MISSING else b, o, t)
        else:
            c = FieldConflict(path, b, o, t)
            try:
                value = _resolve(strategy, c)
                resolved.append(c)
            except MergeConflictError:
                unresolved.append(c)
                continue

        if value is not _MISSING:
            merged[k] = copy.deepcopy(value)

    if unresolved:
        raise MergeConflictError(unresolved)
    return merged, resolved


# ── transcripts ──────────────────────────────────────────────────────────

def common_prefix_len(a: list[dict], b: list[dict]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def safe_cut(messages: list[dict], cut: int) -> int:
    """
    Move a cut point back to the start of a user turn so neither side of the
    cut leaves a tool call without its tool result (providers reject that).
    """
    while cut > 0 and not (messages[cut - 1].get("role") == "assistant"
                           and not messages[cut - 1].get("tool_calls")
                           and (cut == len(messages) or messages[cut].get("role") == "user")):
        cut -= 1
    return cut


def annotate(suffix: list[dict], note: str) -> list[dict]:
    """Label the first user message of a spliced-in suffix (keeps roles valid)."""
    out = copy.deepcopy(suffix)
    for m in out:
        if m.get("role") == "user":
            m["content"] = f"{note} {m.get('content', '')}"
            break
    return out


def merge_transcripts(
    ours: list[dict],
    theirs: list[dict],
    theirs_label: str,
    strategy: str = "append",
    summarizer: Callable[[list[dict]], str] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Merge two transcripts that diverged from a shared history.

    The agent's own line (ours) stays verbatim. What happened on the other
    line (theirs) after the divergence point is folded in:

    - "append": spliced in after the shared history, before ours' new turns,
      with its first user message labelled as a parallel session.
    - "digest": replaced by a short summary returned in `notes`
      (requires `summarizer`), keeping the transcript compact.

    Returns (transcript, notes) — notes are stored in state["_merge_notes"].
    """
    cut = safe_cut(ours, min(common_prefix_len(ours, theirs), len(ours), len(theirs)))
    shared, ours_new, theirs_new = ours[:cut], ours[cut:], theirs[cut:]
    if not theirs_new:
        return list(ours), {}
    if not ours_new:
        return list(theirs), {}

    if strategy == "digest":
        if summarizer is None:
            raise ValueError("transcript strategy 'digest' needs a summarizer")
        return shared + ours_new, {"from": theirs_label, "digest": summarizer(theirs_new),
                                   "turns_folded": sum(m.get("role") == "user" for m in theirs_new)}

    note = f"[Parallel session '{theirs_label}', merged in:]"
    return shared + annotate(theirs_new, note) + ours_new, {
        "from": theirs_label, "turns_appended": sum(m.get("role") == "user" for m in theirs_new)}
