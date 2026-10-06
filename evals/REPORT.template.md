# Agent Registry — Evaluation Report

How these were run, and the fairness notes: [`evals/README.md`](../README.md). Raw per-trial data is in the JSON files next to this report. Machine: 12-core x86-64 Linux laptop, local SSD, Python 3.13.

---

## 1 · Concurrent writers to one agent

### 1a · Instant sessions (pure storage overhead): 20 operations per process, 3 repetitions

{{CONCURRENCY}}

### 1b · Realistic sessions: 500 ms of work per session (a stand-in for an LLM call), 5 operations per process, 2 repetitions

{{CONCURRENCY_LONG}}

**Reading the table**
- **Silent loss** is the claim under test: work reported as saved that the next reader of the agent won't see.
- **Open-source LangGraph** (`langgraph-sqlite`) has no protection when several processes run the same `thread_id`. Each run loads the latest checkpoint and writes a child of it, so concurrent runs produce sibling checkpoints and only one line survives in the state the next run loads. The other events still exist in orphaned checkpoints, so they can be recovered from history, but the agent doesn't see them.
- **`lightx-reject`** (reproducing LangGraph Platform's "reject") never loses work silently, but it pushes failures onto callers, and they grow quickly with concurrency.
- **`lightx-fork`** never loses work either, but it scatters it across side branches that someone must merge later.
- **`lightx-enqueue`** (reproducing LangGraph Platform's "enqueue") keeps everything on main by running sessions one at a time.
  - With instant sessions (1a) it actually has **higher throughput than merge**, because merging costs more than waiting in a queue when there's almost nothing to wait for.
  - With realistic sessions (1b) the queue becomes the bottleneck: every session waits for all the LLM calls queued ahead of it. Throughput stays flat at about 2 ops/s, and p95 latency grows with the number of processes. Merge mode runs sessions in parallel and only serialises the brief commit.
- **`lightx-merge`** keeps every update on main with no failures surfaced to callers, at the cost of merge work during commits.

**A performance bug this evaluation found.** In the first run, `lightx-merge` at 12 processes reached p95 ≈ 2.1 s and 16 ops/s. Finding the common ancestor walked the history depth-first and could re-visit versions many times once the history had many merges. Switching to breadth-first search with cached parent links gave ~176 ops/s and p95 ≈ 130 ms with the same zero loss. The table above is from the fixed code.

---

## 2 · Task success after merging

{{MERGE}}

`scripted` is a perfect-recall reader that answers only from its context. It measures whether the facts **survived into the agent's context**, an upper bound for any model. Rows for real models measure end-to-end success, including whether the model uses the merged context correctly.

**What the results show**
- **Merging recovers the work that's otherwise lost.** Without a merge, the order number learned by session A is gone from the main line (0/30, for any model). With either merge strategy it's back: 30/30 for both the perfect reader and qwen2.5:3b.
- **End-to-end scores are limited by the small model, not by the merge.** qwen2.5:3b restates the address poorly even when nothing was merged (13/30 in `sequential`). When a fact is missing, it often invents placeholders for both fields (`123 Main St, Anytown`), which is why `no-merge` scores 1/30 on the address even though the address is in its context.
- **`digest` scored highest (22/30 vs 13/30 sequential).** A plausible reason: the summary restates facts plainly in the system prompt, which a small model uses more reliably than an earlier exchange. With n=30 the confidence intervals overlap, so treat this as suggestive. Larger models are needed to separate `append` from `digest`.
- **Scoring note:** these numbers were re-scored after a scorer fix (`evals/rescore_merge_task.py`). The model's saved answers are unchanged, and the ground truth was regenerated from the seed. The first scorer rejected correct order numbers written without the `ORD-` prefix.

---

## 3 · Leakage under extraction attacks

{{LEAKAGE}}

`scripted-worst-case` obeys every attack and repeats its whole context, so for it leakage = exposure. That's the most any model could leak.

- **Card number and API key:** redacted execution keeps exposure at 0%. The model never receives them, so no prompt, jailbreak or encoding trick can get them out. This guarantee doesn't depend on the model.
- **Security answer in plain prose:** it is exposed under redaction too. Pattern-based redaction can't recognise free text as secret. That's a real limitation, measured here on purpose. Sensitive facts must go in private state keys, not be left in conversation text.
- **The instruction-only baseline** relies entirely on the model's willingness to refuse. With qwen2.5:3b, a system prompt saying "never reveal" still let the card number out in **35%** of attacks, and the API key in 15%.
- **Redacted execution brought both to 0%**, by construction, for any model.
- **The unprotected security answer leaked about equally in both conditions** (22% vs 13%). That confirms the limitation: prose secrets need `private_keys`, not pattern redaction.

---

## Threats to validity

- **One machine, a local filesystem, and synthetic operations** (an append per session). Real sessions are dominated by LLM latency, which shifts the trade-off between merge and enqueue as described above.
- **The reject and enqueue baselines are reproductions** of LangGraph Platform behaviour on the LightX storage, not the platform itself.
- **Task 2 tests two facts in a short conversation.** Long divergent histories and contradictory updates are harder and aren't covered.
- **Leakage matching** (digits, re-spacing, base64) can miss creative encodings, which undercounts leakage for real models. Exposure is exact.
