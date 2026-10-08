"""Project plumbing for the doctor: findings, file scanning, sandboxed copies, probes."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..sandbox import BubblewrapBackend, Sandbox, SandboxPolicy, SubprocessBackend, auto_backend

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", ".env", "node_modules",
             "build", "dist", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             ".eggs", "site-packages", ".idea", ".vscode"}


@dataclass
class Finding:
    """One problem the doctor found."""

    check: str                    # "dependency" | "import" | "deprecation" | "saved-agent" | "tests"
    severity: str                 # "error" | "warning" | "info"
    title: str
    detail: str = ""
    file: str | None = None       # project-relative path
    line: int | None = None
    fixable: str | None = None    # "code" | "dependency" | "migration" | None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def where(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        return self.file or ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_py_files(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        out += [Path(dirpath) / f for f in filenames if f.endswith(".py")]
    return sorted(out)


def iter_files(root: Path, pattern: str) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        out += [Path(dirpath) / f for f in filenames if Path(f).match(pattern)]
    return sorted(out)


def local_top_names(root: Path) -> set[str]:
    """Top-level importable names provided by the project itself."""
    names = set()
    for base in (root, root / "src"):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.suffix == ".py":
                names.add(p.stem)
            elif p.is_dir() and p.name not in SKIP_DIRS and (
                    (p / "__init__.py").exists() or any(p.glob("*.py"))):
                names.add(p.name)
    return names


@dataclass
class ImportUse:
    module: str
    names: list[str]
    file: str
    line: int


def collect_imports(root: Path) -> list[ImportUse]:
    """Absolute imports of NON-local modules in the project, with locations."""
    local = local_top_names(root)
    uses: list[ImportUse] = []
    for path in iter_py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] not in local:
                        uses.append(ImportUse(a.name, [], rel, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] not in local:
                    names = [a.name for a in node.names if a.name != "*"]
                    uses.append(ImportUse(node.module, names, rel, node.lineno))
    return uses


_PROBE = r'''
import importlib, json, sys, warnings
out = []
for r in json.loads(sys.stdin.read()):
    res = {"module": r["module"], "error": None, "missing": [], "warnings": []}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            mod = importlib.import_module(r["module"])
            for n in r["names"]:
                if not hasattr(mod, n):
                    try:
                        importlib.import_module(r["module"] + "." + n)
                    except Exception:
                        res["missing"].append(n)
        except BaseException as e:
            res["error"] = f"{type(e).__name__}: {e}"
    res["warnings"] = sorted({f"{x.category.__name__}: {x.message}" for x in w
                              if issubclass(x.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning))})
    out.append(res)
print("@@DOCTOR@@" + json.dumps(out))
'''


class Workspace:
    """A disposable copy of the project, run inside the sandbox."""

    def __init__(self, project: Path, *, backend: str = "auto", timeout_s: float = 300,
                 base_dir: Path | None = None):
        self.project = project.resolve()
        base = (base_dir or Path.home() / ".lightx" / "doctor" / "work").expanduser()
        self.root = base / uuid.uuid4().hex[:12]
        self.root.mkdir(parents=True, exist_ok=True)
        extra = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        extra = [p for p in extra if p != self.project and self.project not in p.parents]
        self._extra_pythonpath = extra
        policy = SandboxPolicy(workspace=self.root, read_paths=[Path.home(), *extra, *_python_paths()],
                               allow_network=False, timeout_s=timeout_s, memory_mb=4096)
        be = {"auto": auto_backend, "subprocess": SubprocessBackend,
              "bubblewrap": BubblewrapBackend}[backend]()
        self.sandbox = Sandbox(policy=policy, backend=be)

    def fresh_copy(self, name: str) -> Path:
        dest = self.root / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.project, dest, symlinks=True,
                        ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.egg-info", "*.pyc"))
        return dest

    def env_for(self, copy: Path) -> dict[str, str]:
        import site
        paths = [str(copy)] + ([str(copy / "src")] if (copy / "src").is_dir() else [])
        return {"PYTHONPATH": os.pathsep.join(paths + [str(p) for p in self._extra_pythonpath]),
                "PYTHONDONTWRITEBYTECODE": "1",
                # The sandbox points HOME at the workspace; keep packages installed with
                # `pip install --user` (~/.local/...) visible to the project's code.
                "PYTHONUSERBASE": site.getuserbase()}

    def run(self, copy: Path, argv: list[str], stdin: str | None = None):
        # The sandbox's working directory is the workspace root; cd into the copy.
        quoted = " ".join(_sh_quote(a) for a in argv)
        return self.sandbox.run_argv(["/bin/sh", "-c", f"cd {_sh_quote(str(copy))} && {quoted}"],
                                     stdin=stdin, env=self.env_for(copy))

    def probe_imports(self, copy: Path, uses: list[ImportUse]) -> dict[tuple[str, tuple[str, ...]], dict]:
        """Import each external module inside the sandbox; report errors, missing names, warnings."""
        if not uses:
            return {}
        reqs = [{"module": u.module, "names": u.names} for u in uses]
        result = self.run(copy, [sys.executable, "-c", _PROBE], stdin=json.dumps(reqs))
        marker = next((l for l in result.stdout.splitlines() if l.startswith("@@DOCTOR@@")), None)
        if marker is None:
            raise RuntimeError(f"Import probe failed:\n{result.to_text(4000)}")
        return {(r["module"], tuple(u.names)): r for r, u in zip(json.loads(marker[10:]), uses)}

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _python_paths() -> list[Path]:
    """Everything the project's Python needs to be visible (read-only) in the sandbox:
    the environment, the real interpreter it links to, and its package folders."""
    import site
    paths = {Path(sys.prefix), Path(sys.base_prefix),
             Path(os.path.realpath(sys.executable)).parent.parent}
    try:
        paths |= {Path(p) for p in site.getsitepackages()}
    except AttributeError:  # some virtualenv builds
        pass
    paths.add(Path(site.getuserbase()))
    return sorted({p.resolve() for p in paths if p.exists()})


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ── where did a missing name / module go? (static search, no imports) ─────

def _package_dir(top: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec and spec.origin and spec.origin.endswith(".py"):
        return Path(spec.origin)
    return None


def find_definitions(top_package: str, name: str, limit: int = 8) -> list[str]:
    """Modules under `top_package` (and sibling packages like top_package_*) defining `name`."""
    rx = re.compile(rf"^(?:\s*(?:async\s+)?def|\s*class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*=",
                    re.MULTILINE)
    hits: list[str] = []
    for pkg in _related_packages(top_package):
        base = _package_dir(pkg)
        if base is None or not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            try:
                if rx.search(f.read_text(encoding="utf-8", errors="ignore")):
                    rel = f.relative_to(base.parent).with_suffix("")
                    parts = [p for p in rel.parts if p != "__init__"]
                    hits.append(".".join(parts))
            except OSError:
                continue
            if len(hits) >= limit:
                return hits
    return hits


def find_module_candidates(module: str, limit: int = 8) -> list[str]:
    """For a missing module a.b.c, installed modules ending in .b.c or .c under related packages."""
    parts = module.split(".")
    tail = parts[1:] or parts
    hits = []
    for pkg in _related_packages(parts[0]):
        base = _package_dir(pkg)
        if base is None or not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            rel = f.relative_to(base.parent).with_suffix("")
            dotted = [p for p in rel.parts if p != "__init__"]
            if dotted[-len(tail):] == tail or dotted[-1:] == tail[-1:]:
                hits.append(".".join(dotted))
                if len(hits) >= limit:
                    return hits
    return hits


def _related_packages(top: str) -> list[str]:
    """`top` plus installed siblings like top_core / top_community (common after splits)."""
    found = [top]
    seen = {top}
    for entry in sys.path:
        p = Path(entry)
        if not p.is_dir():
            continue
        try:
            for child in p.iterdir():
                n = child.name
                if (n.startswith(top + "_") or n.startswith(top + "-")) and n not in seen and (
                        child.is_dir() and (child / "__init__.py").exists()):
                    seen.add(n)
                    found.append(n)
        except OSError:
            continue
    return found


def line_excerpt(path: Path, line: int | None, radius: int = 15) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if line is None:
        lo, hi = 0, min(len(lines), 2 * radius)
    else:
        lo, hi = max(0, line - 1 - radius), min(len(lines), line + radius)
    return "\n".join(f"{i + 1:>4}| {lines[i]}" for i in range(lo, hi))
