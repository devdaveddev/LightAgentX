"""Keep the evaluation harness working (LLM-free modes, small sizes)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args, out):
    r = subprocess.run([sys.executable, str(ROOT / "evals" / script), *args, "--out", str(out)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_merge_task_scripted(tmp_path):
    out = run("eval_merge_task.py", "--trials", "3", out=tmp_path / "m.json")
    assert "sequential  both facts: 100%" in out
    assert "no-merge    both facts: 0%" in out
    assert "append      both facts: 100%" in out


def test_leakage_scripted_bound(tmp_path):
    out = run("eval_leakage.py", "--trials", "1", out=tmp_path / "l.json")
    lines = {tuple(l.split()[:2]): l for l in out.splitlines() if l.startswith("  ")}
    assert "exposure 0%" in lines[("redacted", "card")]
    assert "exposure 0%" in lines[("redacted", "api_key")]
    assert "exposure 100%" in lines[("instruction", "card")]


def test_concurrency_lightx_systems(tmp_path):
    out = run("eval_concurrency.py", "--procs", "3", "--ops", "5", "--reps", "1",
              "--systems", "lightx-merge", "lightx-reject", out=tmp_path / "c.json")
    merge = next(l for l in out.splitlines() if l.startswith("lightx-merge"))
    assert "main=15/15" in merge and "silent_loss=0" in merge
    reject = next(l for l in out.splitlines() if l.startswith("lightx-reject"))
    assert "silent_loss=0" in reject


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_concurrency_langgraph_baseline_runs(tmp_path):
    out = run("eval_concurrency.py", "--procs", "1", "--ops", "3", "--reps", "1",
              "--systems", "langgraph-sqlite", out=tmp_path / "c.json")
    assert "main=3/3" in out
