"""One worker process for evals/eval_concurrency.py.

Each operation = load the shared agent's state, append one unique event,
persist. Prints a JSON line with per-operation latency and outcome.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Simulated in-session work (e.g. an LLM call) between loading and saving.
WORK_S = float(os.environ.get("EVAL_WORK_MS", "0")) / 1000


def run_lightx(root: str, agent_id: str, worker: str, n: int, mode: str) -> list[dict]:
    from lightagentx.state import AgentRegistry, ConflictError
    from lightagentx.state.store import FileLock

    reg = AgentRegistry(root)
    wait_for_start()
    ops = []
    for i in range(n):
        event = f"{worker}-{i}"
        t0 = time.perf_counter()
        outcome = "ok"
        try:
            if mode == "enqueue":
                # Reproduces "enqueue": whole sessions run one at a time.
                with FileLock(Path(root) / "queue.lock", timeout_s=3600):
                    with reg.attach(agent_id, "alice", on_conflict="reject") as s:
                        time.sleep(WORK_S)
                        s.state.setdefault("events", []).append(event)
            else:
                with reg.attach(agent_id, "alice", on_conflict=mode) as s:
                    time.sleep(WORK_S)
                    s.state.setdefault("events", []).append(event)
                    branch_before = s.branch
                if mode == "fork" and s.branch != branch_before:
                    outcome = "forked"
        except ConflictError:
            outcome = "rejected"
        ops.append({"event": event, "latency_s": time.perf_counter() - t0, "outcome": outcome})
    return ops


def build_langgraph(db: str):
    """The baseline: one node, state merged by LangGraph's own list reducer."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    from evals.lg_state import EventsState

    g = StateGraph(EventsState)
    def agent(state):
        time.sleep(WORK_S)
        return {}  # the input event is appended by the reducer

    g.add_node("agent", agent)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    conn = sqlite3.connect(db, check_same_thread=False, timeout=60)
    return g.compile(checkpointer=SqliteSaver(conn)), conn


LG_CONFIG = {"configurable": {"thread_id": "shared-agent"}}


def run_langgraph(db: str, worker: str, n: int) -> list[dict]:
    graph, conn = build_langgraph(db)
    wait_for_start()
    ops = []
    for i in range(n):
        event = f"{worker}-{i}"
        t0 = time.perf_counter()
        outcome = "ok"
        try:
            graph.invoke({"events": [event]}, LG_CONFIG)
        except Exception as e:  # e.g. sqlite "database is locked"
            outcome = f"error:{type(e).__name__}"
        ops.append({"event": event, "latency_s": time.perf_counter() - t0, "outcome": outcome})
    conn.close()
    return ops


def wait_for_start() -> None:
    """All workers begin at the same instant, after their (slow) imports."""
    start = float(os.environ.get("EVAL_START_AT", "0"))
    while time.time() < start:
        time.sleep(0.001)


def main() -> None:
    system, target, worker, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if system == "langgraph-sqlite":
        ops = run_langgraph(target, worker, n)
    else:
        agent_id = sys.argv[5]
        ops = run_lightx(target, agent_id, worker, n, system.removeprefix("lightx-"))
    print(json.dumps({"worker": worker, "pid": os.getpid(), "ops": ops}))


if __name__ == "__main__":
    main()
