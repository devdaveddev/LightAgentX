"""
Evaluation 1 — lost updates and latency when independent processes write to
the same agent at the same time.

Systems:
  lightx-merge      AgentRegistry, on_conflict="merge" (three-way merge)
  lightx-fork       AgentRegistry, on_conflict="fork"  (conflicting work -> side branch)
  lightx-reject     AgentRegistry, on_conflict="reject" — reproduces LangGraph Platform "reject"
  lightx-enqueue    sessions serialized by a global lock — reproduces LangGraph Platform "enqueue"
  langgraph-sqlite  real open-source LangGraph + SqliteSaver checkpointer, one shared thread_id

LangGraph's reject/enqueue are features of its hosted platform, not the open-source
library, so they are reproduced here on the same storage as the LightX runs.

Usage:  python evals/eval_concurrency.py [--procs 1 2 4 8 12] [--ops 20] [--reps 3]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKER = Path(__file__).with_name("concurrency_worker.py")
RESULTS = Path(__file__).with_name("results")

SYSTEMS = ["lightx-merge", "lightx-fork", "lightx-reject", "lightx-enqueue", "langgraph-sqlite"]


def final_events(system: str, target: str, agent_id: str | None) -> tuple[set[str], set[str]]:
    """(events on the main line, events preserved anywhere incl. side branches)."""
    if system == "langgraph-sqlite":
        from evals.concurrency_worker import LG_CONFIG, build_langgraph
        graph, conn = build_langgraph(target)
        main = set(graph.get_state(LG_CONFIG).values.get("events", []))
        anywhere: set[str] = set()  # events surviving in ANY checkpoint of the thread
        for cp in graph.get_state_history(LG_CONFIG):
            anywhere |= set(cp.values.get("events", []))
        conn.close()
        return main, anywhere

    from lightagentx.state import AgentRegistry
    reg = AgentRegistry(target)
    main = set(reg.read(agent_id, "alice", "main")["state"].get("events", []))
    anywhere = set(main)
    for branch in reg.branches(agent_id, "alice"):
        anywhere |= set(reg.read(agent_id, "alice", branch)["state"].get("events", []))
    return main, anywhere


def run_once(system: str, procs: int, ops: int, work_ms: float = 0) -> dict:
    with tempfile.TemporaryDirectory(prefix="lx-eval-") as tmp:
        return _run_once(system, procs, ops, work_ms, Path(tmp))


def _run_once(system: str, procs: int, ops: int, work_ms: float, tmp: Path) -> dict:
    agent_id = None
    if system == "langgraph-sqlite":
        target = str(tmp / "checkpoints.sqlite")
    else:
        from lightagentx.state import AgentRegistry
        target = str(tmp / "registry")
        agent_id = AgentRegistry(target).create("Shared", owner="alice", schema={"events": "list"})

    env = {**os.environ, "EVAL_START_AT": str(time.time() + 1.5 + 0.05 * procs),
           "EVAL_WORK_MS": str(work_ms)}
    cmd = lambda w: [sys.executable, str(WORKER), system, target, f"w{w}", str(ops)] + (
        [agent_id] if agent_id else [])
    t0 = time.time()
    ps = [subprocess.Popen(cmd(w), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
          for w in range(procs)]
    outs = []
    for p in ps:
        out, err = p.communicate(timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"{system} worker failed:\n{err[-2000:]}")
        outs.append(json.loads(out.strip().splitlines()[-1]))
    start_at = float(env["EVAL_START_AT"])
    wall = time.time() - max(start_at, t0)

    all_ops = [op for o in outs for op in o["ops"]]
    attempted = {op["event"] for op in all_ops}
    reported_ok = {op["event"] for op in all_ops if op["outcome"] == "ok"}
    forked = {op["event"] for op in all_ops if op["outcome"] == "forked"}
    surfaced = {op["event"] for op in all_ops if op["outcome"] not in ("ok", "forked")}
    main, anywhere = final_events(system, target, agent_id)

    lat = sorted(op["latency_s"] for op in all_ops)
    return {
        "system": system, "procs": procs, "ops_per_proc": ops, "work_ms": work_ms,
        "attempted": len(attempted),
        "on_main": len(main & attempted),
        "preserved_anywhere": len(anywhere & attempted),
        "surfaced_failures": len(surfaced),
        "forked": len(forked),
        # Reported as saved, yet the next reader of the agent won't see it:
        # "ok" events missing from the main line, or forked events found nowhere.
        "silent_loss": len(reported_ok - main) + len(forked - anywhere),
        "p50_ms": 1000 * statistics.median(lat),
        "p95_ms": 1000 * lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "throughput_ops_s": len(all_ops) / wall if wall > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--ops", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--work-ms", type=float, default=0,
                    help="Simulated work per session (e.g. LLM latency), in ms.")
    ap.add_argument("--out", type=Path, default=RESULTS / "concurrency.json")
    args = ap.parse_args()

    rows = []
    for procs in args.procs:
        for system in args.systems:
            for rep in range(args.reps):
                r = run_once(system, procs, args.ops, args.work_ms)
                r["rep"] = rep
                rows.append(r)
                print(f"{system:<17} procs={procs:<3} rep={rep}  main={r['on_main']}/{r['attempted']}  "
                      f"silent_loss={r['silent_loss']}  surfaced={r['surfaced_failures']}  "
                      f"forked={r['forked']}  p50={r['p50_ms']:.1f}ms  p95={r['p95_ms']:.1f}ms  "
                      f"{r['throughput_ops_s']:.0f} ops/s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
