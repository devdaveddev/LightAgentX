# Agent Registry — Versioned, Permissioned Agent State Across Processes

`lightagentx.state` turns an agent from an in-memory object into a **persistent, addressable, versioned resource** that independent processes can discover, attach to, continue, fork and merge, under per-principal access control.

```python
from lightagentx import AgentRegistry, AccessPolicy

reg = AgentRegistry("~/.lightx/agents")          # a directory any process can point at
aid = reg.create("SupportBot", owner="alice",
                 policy=AccessPolicy(owner="alice", private_keys={"card_number"}),
                 schema={"customer": "str", "decisions": "list", "card_number": "str"})

# Process A (today)
with reg.attach(aid, "alice", llm=llm, workflow="intake") as s:
    s.run("Customer ACME wants a refund for order 17")
    s.state["customer"] = "ACME"
# -> new immutable version, with provenance

# Process B (tomorrow, a different program) — knows only the name
aid = reg.find("SupportBot")
with reg.attach(aid, "alice", llm=llm, workflow="followup") as s:
    s.run("What did we decide?")        # the LLM receives A's actual conversation
```

---

## The five questions

| # | Question | Answer | Mechanism |
|---|---|---|---|
| 1 | Does an agent persist after its original process ends? | **Yes** | Every session commits a version to disk; nothing lives only in memory |
| 2 | Can another independent process discover and access it? | **Yes** | `list_agents()` / `find(name)` / stable `agent_id`, filtered by the caller's rights |
| 3 | Does it retrieve the agent's actual prior state, not a RAG of old messages? | **Yes** | The exact transcript (tool calls included, byte-for-byte) plus typed structured state are loaded into the agent |
| 4 | Can the agent continue execution from that state? | **Yes** | `attach()` returns a live `SingleAgent` whose memory *is* the stored state |
| 5 | Isolation, conflicts, versioning and permissions? | **Yes** (see below) | Content-addressed versions, compare-and-swap branches, three-way merge, ACLs, redaction, audit |

All five are verified by tests that use **separate OS processes**, not threads or mocks (`tests/test_registry.py::TestAcrossProcesses`).

---

## Architecture

```
root/
├── objects/ab/cdef….json        immutable versions, named by SHA-256 of their content
└── agents/<agent_id>/
    ├── meta.json                name, owner, schema, access policy
    ├── refs/<branch>            the version id a branch points to
    ├── audit.jsonl              append-only: who did what, allowed/denied
    └── .lock                    cross-process lock for branch moves
```

### Versions (`store.py`)

A version is like a git commit. It holds the transcript, the structured state, the config, the tool manifest, its **parents**, and **provenance**: principal, workflow, pid, host, OS user and program. Its id is the SHA-256 of all that, so:

- versions can never change after they're written;
- `verify()` re-hashes the whole history, so editing any stored version is detected (tamper-evident; a test covers this);
- history is a DAG, which allows time travel (`at=<version>`), `log`, `diff` and `merge_base`.

Provenance is **asserted, not signed**: it records who claims to have done something, and the hash chain makes later tampering detectable.

### Sessions and concurrency (`registry.py`)

`attach()` loads a branch head into a live agent. When the block exits cleanly, the session commits. If it raised, the work is discarded and the abort is logged.

A commit moves the branch by **compare-and-swap** under a per-agent file lock: it only lands if the branch still points at the version the session started from. If another process got there first, the session's `on_conflict` decides what happens:

| `on_conflict` | Behaviour |
|---|---|
| `"merge"` (default) | Three-way merge of the session's version with the new head; the merged version gets two parents |
| `"fork"` | Commit to a new branch `main.fork-<id>`; main is untouched |
| `"reject"` | Raise `ConflictError` |

Commits first try a few lock-free attempts. Under contention they fall back to doing read, merge and swap while holding the lock, so a `"merge"` commit **always** lands. A stress test runs 12 processes × 20 commits against one agent and checks that all 240 updates survive, with nothing lost or duplicated. An earlier optimistic-only version failed this test; that's why the fallback exists.

### Merging (`merge.py`)

- **Structured state** uses a recursive three-way merge against the common ancestor:
  - a key changed on one side takes that side;
  - dicts merge recursively;
  - lists merge order-preservingly (both sides' additions kept, removals respected);
  - a true conflict uses `merge_strategy`: `"raise"`, `"ours"`, `"theirs"`, or a function. Resolved conflicts are recorded in `state["_merge_notes"]`.
- **Transcripts** are where the open problem is. Two strategies:
  - `"append"`: the other line's new turns are spliced in after the shared history, with their first user message labelled `[Parallel session '…', merged in:]`. The cut point is moved back to a turn boundary, so no tool call is ever separated from its result. Every provider accepts the result, and a test checks this.
  - `"digest"`: the other line's turns are replaced by a summary (your `summarizer`, typically an LLM call), which is added to the system prompt on the next attach.

### Permissions (`access.py`)

Rights: `read`, `run`, `write`, `fork`, `merge`, `private`, `admin`.
- Grants are per principal, and patterns are allowed (`"billing-*"`).
- Writers can be limited to certain branches. A writer outside its branches, but with `fork`, lands on a personal branch `<principal>.<branch>`.
- `run` without `write` means the agent can be used, but nothing it does is saved.

**Redacted execution with lossless write-back.** A principal without `private` attaches to a *redacted* copy: private state keys are removed, and secrets in the transcript (API keys, tokens, private keys, `password=…`) become `[REDACTED]`. The agent's LLM therefore never receives the hidden data and can't leak it.

When that session commits, only its **changes** are written back onto the full, unredacted version: new transcript turns are appended, and only the state keys it changed are updated. Hidden data survives unchanged. Trying to write a private key without the right raises `AccessDenied`.

Every access decision, including denials, is written to `audit.jsonl`. Only `admin` can read it.

---

## Limitations (honest)

- **Principals are names, not authenticated identities.** Anyone who can write to the registry directory can claim any principal. This prototype is for cooperating processes on one machine or shared filesystem; real authentication needs a server in front of it.
- **The storage backend is the local filesystem** (`fcntl` locks; `msvcrt` on Windows, untested). Network filesystems with unreliable locking aren't supported.
- **Transcript merge is a heuristic.** Splicing keeps the transcript valid and labelled, but it can't make two divergent conversations semantically consistent. `"digest"` trades fidelity for coherence.
- **Redaction is pattern-based.** It catches common secret formats, not arbitrary sensitive text. For guaranteed hiding, use `private_keys` on structured state.
- **History grows without limit.** There's no garbage collection or compaction yet, and full ancestry walks are O(history).
- **Only `SingleAgent` is supported**, and `SummaryMemory` isn't preserved as such: transcripts are stored in full.

---

## Related work

| System | Persistent across processes | Branch / fork | Merge of divergent state | Concurrency policy | Access control / redaction | Provenance / integrity |
|---|---|---|---|---|---|---|
| [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers.md) | ✅ by `thread_id` | ✅ resuming an old checkpoint forks ([time travel](https://docs.langchain.com/oss/javascript/langgraph/use-time-travel)) | ❌ | Server only: reject / enqueue / interrupt / rollback ([double texting](https://docs.langchain.com/langsmith/double-texting)) | ❌ in the library | ❌ |
| [Letta](https://docs.letta.com/guides/core-concepts/stateful-agents/index.md) | ✅ server-hosted agents with ids | export/import via [Agent File](https://docs.letta.com/guides/agents/agent-file) | ❌ ([shared blocks](https://docs.letta.com/guides/agents/multi-agent-shared-memory) instead) | not documented | ❌ | ❌ |
| [AgentGit](https://arxiv.org/abs/2511.00628) (on LangGraph) | via LangGraph | ✅ commit / revert / branch | not described | not described | ❌ | ❌ |
| [Git Context Controller](https://arxiv.org/abs/2508.00031) | ✅ files in git | ✅ | ✅ by LLM summary synthesis | ❌ | ❌ | git only |
| [GitOfThoughts](https://arxiv.org/abs/2606.14470) | ✅ git repo | ✅ | ✅ cross-agent, reasoning trees | ❌ | ❌ | ✅ content-addressed, signed commits |
| [Collaborative Memory](https://www.researchgate.net/publication/392106132_Collaborative_Memory_Multi-User_Memory_Sharing_in_LLM_Agents_with_Dynamic_Access_Control) | ✅ (fact fragments) | ❌ | ❌ | ❌ | ✅ dynamic access graph, read/write transforms | ✅ provenance attributes |
| [Governed Shared Memory](https://arxiv.org/abs/2606.24535) | ✅ (knowledge store) | ❌ | supersession, not merge | partial | ✅ scoped retrieval, policies | ✅ |
| [MAP-Graph](https://arxiv.org/abs/2608.10509) | ✅ (memory graph) | ❌ | ❌ ("not merged") | ❌ | ✅ permission filtering, REDACT decisions | ✅ provenance graph, content hashes |
| [AIM](https://arxiv.org/abs/2609.12320) | ✅ (fact memory) | ❌ | ❌ | ❌ | ✅ private/shared scopes | ❌ |
| **LightAgentX registry** | ✅ | ✅ | ✅ three-way state + transcript | ✅ CAS + merge/fork/reject, proven under 12-process load | ✅ ACLs, branch scopes, redacted execution with lossless write-back | ✅ content-addressed DAG, asserted provenance, audit |

Each cell is based on the cited documentation or paper text as read on 2026-10-05; "not described" means the source doesn't say.
