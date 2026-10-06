"""Re-score saved eval-2 answers with the current scorer (no model calls).

Ground truth is regenerated from the run's seed, so results saved by an older
scorer can be re-scored exactly. Answers are never changed.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.eval_merge_task import make_trial, score  # noqa: E402

for path in sys.argv[1:]:
    p = Path(path)
    d = json.loads(p.read_text())
    rng = random.Random(d["seed"])
    trials = [make_trial(rng) for _ in range(d["trials"])]
    conds = list(dict.fromkeys(r["condition"] for r in d["rows"]))
    rescored = []
    for ci, cond in enumerate(conds):
        rows = [r for r in d["rows"] if r["condition"] == cond]
        assert len(rows) == len(trials), (cond, len(rows))
        for r, t in zip(rows, trials):
            rescored.append({"condition": cond, **t, "answer": r["answer"], **score(r["answer"], t)})
    d["rows"] = rescored
    d["rescored"] = "answers unchanged; truth regenerated from seed; scorer accepts the number without the 'ORD-' prefix"
    p.write_text(json.dumps(d, indent=2))
    for cond in conds:
        got = [r for r in rescored if r["condition"] == cond]
        print(f"{p.name}  {cond:<11} both={sum(r['both'] for r in got)}/{len(got)}  "
              f"order={sum(r['order_ok'] for r in got)}  address={sum(r['address_ok'] for r in got)}")
