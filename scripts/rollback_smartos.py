#!/usr/bin/env python3
"""
Completely remove SmartOS (lightagentx.sandbox + lightagentx.smartos) from this repo.

How it works:
  1. Finds every commit that belongs to SmartOS — commits carrying the
     ``SmartOS-Rollback-Group: smartos`` trailer, plus any commit that touched
     SmartOS-only paths (so later follow-up work is caught too).
  2. Reverts them newest-first into ONE new commit. History is preserved, so
     the rollback itself can be undone with ``git revert <that commit>``.
  3. Verifies the package still imports and no SmartOS files remain.

Nothing outside SmartOS is touched — e.g. the OpenAI base_url fix lives in its
own commit and stays.

Usage:
    python scripts/rollback_smartos.py --dry-run      # show what would happen
    python scripts/rollback_smartos.py                # revert (asks first)
    python scripts/rollback_smartos.py --yes          # revert without asking
    python scripts/rollback_smartos.py --purge-data   # also delete ~/.lightx workspace + audit log

Alternatives if SmartOS was never merged to main:
    git checkout main && git branch -D feature/smartos
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TRAILER = "SmartOS-Rollback-Group: smartos"

SMARTOS_PATHS = [
    "lightagentx/sandbox",
    "lightagentx/smartos",
    "tests/test_sandbox.py",
    "tests/test_smartos.py",
    "examples/08_smart_os.py",
    "docs/smartos-build-log.md",
    "scripts/rollback_smartos.py",
]


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def find_smartos_commits() -> list[str]:
    """SmartOS commits on the current branch, newest first."""
    tagged = git("log", "--no-merges", "--format=%H", f"--grep={TRAILER}", "--fixed-strings").split()
    by_path = git("log", "--no-merges", "--format=%H", "--", *SMARTOS_PATHS).split()
    wanted = set(tagged) | set(by_path)
    ordered = git("log", "--no-merges", "--format=%H").split()
    return [c for c in ordered if c in wanted]


def purge_user_data(include_trash: bool) -> None:
    root = Path.home() / ".lightx"
    for name in ("workspace", "audit.jsonl"):
        p = root / name
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  removed {p}")
        elif p.exists():
            p.unlink()
            print(f"  removed {p}")
    trash = root / "trash"
    if trash.exists():
        if include_trash:
            shutil.rmtree(trash)
            print(f"  removed {trash}")
        else:
            print(f"  kept {trash} — it holds files SmartOS 'deleted' for you. "
                  f"Recover anything you need, then rerun with --purge-trash.")
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove SmartOS from this repository.")
    ap.add_argument("--dry-run", action="store_true", help="Only show what would be reverted.")
    ap.add_argument("--yes", action="store_true", help="Don't ask for confirmation.")
    ap.add_argument("--purge-data", action="store_true",
                    help="Also delete ~/.lightx/workspace and ~/.lightx/audit.jsonl.")
    ap.add_argument("--purge-trash", action="store_true",
                    help="With --purge-data, also delete ~/.lightx/trash (files SmartOS deleted).")
    args = ap.parse_args()

    repo = Path(git("rev-parse", "--show-toplevel"))
    if git("status", "--porcelain"):
        raise SystemExit("Working tree has uncommitted changes. Commit or stash them first.")

    commits = find_smartos_commits()
    if not commits:
        print("No SmartOS commits found on this branch — nothing to roll back.")
        if args.purge_data and not args.dry_run:
            purge_user_data(args.purge_trash)
        return 0

    print(f"Branch: {git('branch', '--show-current') or '(detached)'}")
    print("SmartOS commits to revert (newest first):")
    for c in commits:
        print(f"  {git('log', '-1', '--format=%h %s', c)}")
    if args.dry_run:
        print("\nDry run — nothing changed.")
        return 0
    if not args.yes and input("\nRevert these into one commit? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    for c in commits:
        r = subprocess.run(["git", "revert", "--no-commit", c], capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["git", "revert", "--abort"], capture_output=True)
            raise SystemExit(
                f"Revert of {c[:9]} conflicted with later changes:\n{r.stderr.strip()}\n"
                f"Nothing was changed. Resolve manually with: git revert {c[:9]}"
            )

    short = ", ".join(c[:9] for c in commits)
    git("commit", "-m", f"Remove SmartOS (rollback)\n\nReverts: {short}\n"
                        f"Undo this rollback with: git revert HEAD")
    print(f"\nCommitted rollback: {git('log', '-1', '--format=%h')}")

    leftovers = [p for p in SMARTOS_PATHS if (repo / p).exists()]
    check = subprocess.run([sys.executable, "-c", "import lightagentx"], cwd=repo,
                           capture_output=True, text=True)
    print("Leftover SmartOS files:", leftovers or "none")
    print("Package import:", "ok" if check.returncode == 0 else f"FAILED\n{check.stderr}")

    if args.purge_data:
        print("Purging user data:")
        purge_user_data(args.purge_trash)

    print("\nIf you installed with pip, reinstall to drop the `lightx-os` command: pip install -e .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
