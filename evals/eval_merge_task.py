"""
Evaluation 2 — does the agent still know what happened in BOTH parallel
sessions after they are combined?

Per trial: an agent has a shared history. Two sessions attach to the same
version at the same time. Session A learns the order number; session B
learns the new delivery address. Then a fresh session asks for both.

Conditions:
  sequential  B attaches after A committed (no conflict)      — upper bound
  no-merge    B's work is forked aside (on_conflict="fork")   — what "reject"/fork leaves on main
  append      three-way merge, transcript_strategy="append"
  digest      three-way merge, transcript_strategy="digest" (LLM-written summary)

--provider scripted uses a perfect-recall reader instead of an LLM: it measures
whether the facts *survived into the context* (an upper bound for any model).
Use a real provider to measure end-to-end task success.

Usage:
  python evals/eval_merge_task.py --trials 30                       # LLM-free
  python evals/eval_merge_task.py --provider openai --model gpt-4o-mini --trials 30
  python evals/eval_merge_task.py --base-url http://localhost:11434/v1 --provider openai --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import (RESULTS, PerfectReaderLLM, add_llm_args, fmt_rate, make_llm,  # noqa: E402
                          model_label)
from lightagentx.state import AgentRegistry  # noqa: E402

CONDITIONS = ["sequential", "no-merge", "append", "digest"]
STREETS = ["Larch", "Copper", "Juniper", "Harbour", "Quarry", "Bramble", "Fennel", "Orchard",
           "Saltmarsh", "Thistle", "Willow", "Kestrel", "Marigold", "Ember", "Cobalt"]
CITIES = ["Leeds", "Galway", "Utrecht", "Porto", "Tartu", "Aarhus", "Ghent", "Bergen", "Lyon"]
PROBE = ("Without asking me anything, reply with exactly one line in this format: "
         "ORDER=<my order number>; ADDRESS=<my current delivery address>")
SYSTEM = "You are a customer-support agent for an online shop. Be brief."
EXTRACT = {"ORDER": r"(ORD-\d{5})", "ADDRESS": r"(\d{1,3} [A-Z][a-z]+ (?:Street|Road|Lane), [A-Z][a-z]+)"}


def make_trial(rng: random.Random) -> dict:
    return {
        "order": f"ORD-{rng.randint(10000, 99999)}",
        "address": f"{rng.randint(2, 199)} {rng.choice(STREETS)} {rng.choice(['Street', 'Road', 'Lane'])}, "
                   f"{rng.choice(CITIES)}",
        "customer": rng.choice(["Dana", "Ravi", "Mei", "Tomas", "Aisha", "Owen", "Lena"]),
    }


def summarizer_for(llm):
    def summarize(messages: list[dict]) -> str:
        convo = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in messages
                          if m["role"] in ("user", "assistant"))
        return llm.chat([{"role": "user", "content":
                          "Summarize this support conversation in 1-2 sentences. Keep every "
                          f"identifier, number and address exactly as written:\n\n{convo}"}]).content.strip()
    return summarize


def score(answer: str, t: dict) -> dict:
    """Credit the right facts regardless of formatting (e.g. '52445' for 'ORD-52445')."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    a = norm(answer)
    order_ok = t["order"].split("-")[1] in re.sub(r"\D", " ", answer).split()
    street = norm(t["address"].split(",")[0])  # house number + street name + type
    address_ok = street in a
    return {"order_ok": order_ok, "address_ok": address_ok, "both": order_ok and address_ok}


def run_trial(condition: str, t: dict, llm, root: Path) -> dict:
    reg = AgentRegistry(root)
    aid = reg.create("Support", owner="alice", system_prompt=SYSTEM, transcript=[
        {"role": "user", "content": f"Hi, I'm {t['customer']}. I have a question about a delivery."},
        {"role": "assistant", "content": f"Hello {t['customer']}, happy to help. What do you need?"},
    ])
    a_msg = f"My order number is {t['order']}. Can you check it?"
    b_msg = f"Please change my delivery address to {t['address']}."

    if condition == "sequential":
        with reg.attach(aid, "alice", llm=llm, workflow="A") as a:
            a.run(a_msg)
        with reg.attach(aid, "alice", llm=llm, workflow="B") as b:
            b.run(b_msg)
    else:
        mode = {"no-merge": "fork", "append": "merge", "digest": "merge"}[condition]
        # B lands first, so A's commit is the one that hits the conflict:
        # the condition's settings belong on A.
        a = reg.attach(aid, "alice", llm=llm, workflow="A", on_conflict=mode,
                       transcript_strategy="digest" if condition == "digest" else "append",
                       summarizer=summarizer_for(llm) if condition == "digest" else None)
        b = reg.attach(aid, "alice", llm=llm, workflow="B")
        a.run(a_msg)
        b.run(b_msg)
        b.commit("B")
        a.commit("A")

    with reg.attach(aid, "alice", llm=llm, workflow="probe") as p:
        answer = p.run(PROBE)
    return {"condition": condition, **t, "answer": answer, **score(answer, t)}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_llm_args(ap)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--out", type=Path, help="Results file (default: evals/results/...).")
    args = ap.parse_args()

    llm = None if args.provider == "scripted" else make_llm(args)
    label = model_label(args, llm)
    rng = random.Random(args.seed)
    trials = [make_trial(rng) for _ in range(args.trials)]
    rows = []
    for cond in args.conditions:
        if cond == "digest" and llm is None:
            print("digest: skipped in scripted mode (needs an LLM to write the summary)")
            continue
        for t in trials:
            model = llm or PerfectReaderLLM(EXTRACT)
            with tempfile.TemporaryDirectory(prefix="lx-merge-") as tmp:
                rows.append(run_trial(cond, t, model, Path(tmp)))
        got = [r for r in rows if r["condition"] == cond]
        print(f"{cond:<11} both facts: {fmt_rate(sum(r['both'] for r in got), len(got))}   "
              f"order: {sum(r['order_ok'] for r in got)}/{len(got)}   "
              f"address: {sum(r['address_ok'] for r in got)}/{len(got)}", flush=True)

    out = args.out or RESULTS / f"merge_task__{label.replace(':', '_').replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": label, "trials": args.trials, "seed": args.seed,
                               "rows": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
