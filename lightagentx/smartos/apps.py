"""App tools — discover installed applications, launch and close them."""

from __future__ import annotations

import configparser
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..sandbox import Risk, Sandbox, SandboxViolation
from ..tools.base import BaseTool, tool


@dataclass
class AppEntry:
    app_id: str
    name: str
    exec_cmd: str
    comment: str = ""
    source: str = ""


def _desktop_dirs() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    dirs = [data_home / "applications"] + [Path(d) / "applications" for d in data_dirs]
    dirs += [Path("/var/lib/flatpak/exports/share/applications"),
             data_home / "flatpak/exports/share/applications"]
    return [d for d in dirs if d.is_dir()]


def discover_apps(dirs: list[Path] | None = None) -> dict[str, AppEntry]:
    """Find launchable GUI apps. Keyed by lowercase app id."""
    apps: dict[str, AppEntry] = {}
    if sys.platform == "darwin" and dirs is None:
        for root in (Path("/Applications"), Path.home() / "Applications",
                     Path("/System/Applications")):
            for app in root.glob("*.app"):
                apps.setdefault(app.stem.lower(), AppEntry(app.stem, app.stem, "", "", str(app)))
        return apps

    for d in dirs if dirs is not None else _desktop_dirs():
        for f in sorted(d.glob("*.desktop")):
            cp = configparser.RawConfigParser(strict=False, interpolation=None)
            try:
                cp.read(f, encoding="utf-8")
                e = cp["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError):
                continue
            if e.get("Type", "Application") != "Application":
                continue
            if e.get("NoDisplay", "false").lower() == "true" or e.get("Hidden", "false").lower() == "true":
                continue
            app_id = f.stem
            apps.setdefault(app_id.lower(), AppEntry(
                app_id=app_id, name=e.get("Name", app_id), exec_cmd=e.get("Exec", ""),
                comment=e.get("Comment", ""), source=str(f),
            ))
    return apps


def find_app(query: str, apps: dict[str, AppEntry]) -> AppEntry | None:
    q = query.lower().strip()
    if q in apps:
        return apps[q]
    for a in apps.values():
        if a.name.lower() == q:
            return a
    candidates = [a for a in apps.values()
                  if q in a.name.lower() or q in a.app_id.lower().split(".")[-1]]
    return min(candidates, key=lambda a: len(a.name)) if candidates else None


def _launch(app: AppEntry) -> None:
    if sys.platform == "darwin":
        cmd = ["open", "-a", app.name]
    elif shutil.which("gtk-launch") and app.source.endswith(".desktop"):
        cmd = ["gtk-launch", app.app_id]
    else:
        # Strip desktop-entry field codes (%f %U ...) — we never pass arguments.
        cmd = [t for t in shlex.split(app.exec_cmd) if not (t.startswith("%") and len(t) == 2)]
        if not cmd:
            raise OSError(f"No Exec line for {app.name}")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)


def make_app_tools(sandbox: Sandbox) -> list[BaseTool]:
    """Build app-management tools bound to a sandbox."""
    cache: dict[str, dict[str, AppEntry]] = {}

    def apps() -> dict[str, AppEntry]:
        if "apps" not in cache:
            cache["apps"] = discover_apps()
        return cache["apps"]

    @tool
    def list_applications(query: str = "") -> str:
        """List installed desktop applications, optionally filtered by a search term.

        Args:
            query: Text to filter app names/descriptions by.
        """
        q = query.lower()
        rows = [f"{a.name}  [{a.app_id}]" + (f" — {a.comment}" if a.comment else "")
                for a in sorted(apps().values(), key=lambda a: a.name.lower())
                if not q or q in a.name.lower() or q in a.comment.lower() or q in a.app_id.lower()]
        if not rows:
            return f"No installed applications match '{query}'."
        return "\n".join(rows[:80]) + (f"\n... and {len(rows) - 80} more" if len(rows) > 80 else "")

    @tool
    def launch_application(name: str) -> str:
        """Launch an installed desktop application by name or app id.

        Args:
            name: Application name (e.g. 'Firefox', 'Files') or id (e.g. 'org.gnome.Nautilus').
        """
        app = find_app(name, apps())
        if app is None:
            return f"No installed application matches '{name}'. Use list_applications to search."
        trusted = {t.lower() for t in sandbox.policy.trusted_apps}
        risk = Risk.LOW if (app.app_id.lower() in trusted or app.name.lower() in trusted) else Risk.MEDIUM
        sandbox.authorize("launch_app", f"{app.name} [{app.app_id}]", risk)
        _launch(app)
        return f"Launched {app.name}."

    @tool
    def close_application(name: str, force: bool = False) -> str:
        """Close all running processes of an application (asks the user first).

        Args:
            name: Process or application name to close (e.g. 'firefox').
            force: Force-kill instead of a graceful close.
        """
        from .tasks import require_psutil, terminate_pid
        require_psutil()
        import psutil

        needle = name.lower().strip()
        app = find_app(name, apps())
        exe_hint = ""
        if app and app.exec_cmd:
            try:
                exe_hint = Path(shlex.split(app.exec_cmd)[0]).name.lower()
            except (ValueError, IndexError):
                pass
        me = psutil.Process().username()
        matches = []
        for p in psutil.process_iter(["pid", "name", "username", "ppid"]):
            pname = (p.info["name"] or "").lower()
            if p.info["username"] != me:
                continue
            if pname == needle or (exe_hint and pname == exe_hint) or pname.startswith(needle):
                matches.append(p)
        if not matches:
            return f"No running processes of yours match '{name}'."
        # Only signal top-level instances; children exit with their parent.
        pids = {p.pid for p in matches}
        roots = [p for p in matches if p.info["ppid"] not in pids]
        results = []
        for p in roots:
            try:
                results.append(terminate_pid(sandbox, p.pid, force))
            except SandboxViolation as e:
                results.append(f"PID {p.pid}: {e}")
                if "declined" in str(e):
                    break
        return "\n".join(results)

    return [list_applications, launch_application, close_application]
