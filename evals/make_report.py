"""Build evals/results/REPORT.md tables from the saved JSON results."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import RESULTS, fmt_rate  # noqa: E402

SYSTEM_ORDER = ["lightx-merge", "lightx-fork", "lightx-reject", "lightx-enqueue", "langgraph-sqlite"]


def concurrency_table(name: str = "concurrency.json") -> str:
    path = RESULTS / name
    if not path.exists():
        return "_not run_\n"
    rows = json.loads(path.read_text())
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["procs"], r["system"])].append(r)
    out = ["| Processes | System | On main (mean) | Silent loss | Surfaced failures | Forked | p50 ms | p95 ms | ops/s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for procs in sorted({k[0] for k in groups}):
        for system in SYSTEM_ORDER:
            g = groups.get((procs, system))
            if not g:
                continue
            att = g[0]["attempted"]
            m = lambda k: statistics.mean(x[k] for x in g)
            loss = sum(x["silent_loss"] for x in g)
            total = sum(x["attempted"] for x in g)
            out.append(
                f"| {procs} | `{system}` | {m('on_main'):.0f}/{att} | "
                f"**{loss}/{total}** ({100 * loss / total:.0f}%) | {m('surfaced_failures'):.0f} | "
                f"{m('forked'):.0f} | {m('p50_ms'):.1f} | {m('p95_ms'):.0f} | {m('throughput_ops_s'):.0f} |")
    return "\n".join(out) + f"\n\n_{len(rows)} runs; each row averages {len(g)} repetitions; silent loss is summed._\n"


def merge_tables() -> str:
    files = sorted(RESULTS.glob("merge_task__*.json"))
    if not files:
        return "_not run_\n"
    parts = []
    for f in files:
        d = json.loads(f.read_text())
        parts.append(f"**Model: `{d['model']}`** ({d['trials']} trials, seed {d['seed']})\n")
        parts.append("| Condition | Both facts | Order number | New address |\n|---|---|---|---|")
        for cond in ["sequential", "no-merge", "append", "digest"]:
            got = [r for r in d["rows"] if r["condition"] == cond]
            if got:
                n = len(got)
                parts.append(f"| {cond} | {fmt_rate(sum(r['both'] for r in got), n)} | "
                             f"{sum(r['order_ok'] for r in got)}/{n} | {sum(r['address_ok'] for r in got)}/{n} |")
        parts.append("")
    return "\n".join(parts)


def leakage_tables() -> str:
    files = sorted(RESULTS.glob("leakage__*.json"))
    if not files:
        return "_not run_\n"
    parts = []
    for f in files:
        d = json.loads(f.read_text())
        parts.append(f"**Model: `{d['model']}`** ({d['trials']} secret sets × {len(d['attacks'])} attacks)\n")
        parts.append("| Condition | Secret | Exposure (reached the LLM) | Leakage (in the reply) |\n|---|---|---|---|")
        for cond in ["instruction", "redacted"]:
            got = [r for r in d["rows"] if r["condition"] == cond]
            for k in ["card", "api_key", "answer"]:
                parts.append(f"| {cond} | {k} | {fmt_rate(sum(r['exposed'][k] for r in got), len(got))} | "
                             f"{fmt_rate(sum(r['leaked'][k] for r in got), len(got))} |")
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    tables = {
        "CONCURRENCY": concurrency_table(),
        "CONCURRENCY_LONG": concurrency_table("concurrency_long_sessions.json"),
        "MERGE": merge_tables(),
        "LEAKAGE": leakage_tables(),
    }
    template = (Path(__file__).with_name("REPORT.template.md")).read_text()
    for k, v in tables.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    (RESULTS / "REPORT.md").write_text(template)
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
