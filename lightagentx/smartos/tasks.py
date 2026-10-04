"""Task-manager tools — inspect system load, list processes, terminate safely."""

from __future__ import annotations

import os
import time
from datetime import datetime

from ..sandbox import Risk, Sandbox
from ..tools.base import BaseTool, tool

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def require_psutil() -> None:
    if psutil is None:
        raise ImportError(
            "The SmartOS task manager needs psutil. "
            "Install with: pip install lightagentx[os]"
        )


def _gb(n: float) -> str:
    return f"{n / 1024**3:.1f}GB"


def _own_lineage() -> set[int]:
    """PIDs of this process and all its ancestors — never killable."""
    pids = {os.getpid()}
    try:
        p = psutil.Process()
        pids.update(a.pid for a in p.parents())
    except psutil.Error:
        pass
    return pids


def snapshot_processes(sort_by: str = "cpu", limit: int = 15, name_filter: str = "") -> list[dict]:
    """Sample processes over ~0.3s so CPU percentages are meaningful."""
    require_psutil()
    procs = []
    for p in psutil.process_iter(["pid", "name", "username"]):
        try:
            p.cpu_percent(None)
            procs.append(p)
        except psutil.Error:
            continue
    time.sleep(0.3)
    rows = []
    needle = name_filter.lower()
    for p in procs:
        try:
            name = p.info["name"] or ""
            if needle and needle not in name.lower():
                continue
            rows.append({
                "pid": p.pid,
                "name": name,
                "user": p.info["username"] or "?",
                "cpu": p.cpu_percent(None),
                "mem_mb": p.memory_info().rss / 1024**2,
                "status": p.status(),
            })
        except psutil.Error:
            continue
    key = {"cpu": "cpu", "memory": "mem_mb", "mem": "mem_mb", "pid": "pid", "name": "name"}.get(sort_by, "cpu")
    rows.sort(key=lambda r: r[key], reverse=key in ("cpu", "mem_mb"))
    return rows[: max(1, limit)]


def format_processes(rows: list[dict]) -> str:
    lines = [f"{'PID':>7}  {'CPU%':>5}  {'MEM':>8}  {'USER':<10} {'STATUS':<9} NAME"]
    for r in rows:
        lines.append(
            f"{r['pid']:>7}  {r['cpu']:>5.1f}  {r['mem_mb']:>6.0f}MB  "
            f"{r['user'][:10]:<10} {r['status'][:9]:<9} {r['name']}"
        )
    return "\n".join(lines)


def system_overview_text() -> str:
    require_psutil()
    cpu = psutil.cpu_percent(interval=0.3)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    lines = [
        f"CPU: {cpu:.0f}% across {psutil.cpu_count()} cores"
        + (f", load avg {', '.join(f'{x:.2f}' for x in os.getloadavg())}" if hasattr(os, "getloadavg") else ""),
        f"Memory: {_gb(vm.used)} / {_gb(vm.total)} used ({vm.percent:.0f}%), {_gb(vm.available)} available",
        f"Swap: {_gb(sw.used)} / {_gb(sw.total)} ({sw.percent:.0f}%)",
        f"Uptime since: {datetime.fromtimestamp(psutil.boot_time()):%Y-%m-%d %H:%M}",
    ]
    seen = set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen or part.fstype in ("squashfs", "tmpfs", "overlay"):
            continue
        seen.add(part.mountpoint)
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        lines.append(f"Disk {part.mountpoint}: {_gb(u.used)} / {_gb(u.total)} ({u.percent:.0f}%)")
    battery = getattr(psutil, "sensors_battery", lambda: None)()
    if battery is not None:
        lines.append(f"Battery: {battery.percent:.0f}% ({'charging' if battery.power_plugged else 'on battery'})")
    return "\n".join(lines)


def terminate_pid(sandbox: Sandbox, pid: int, force: bool = False) -> str:
    """Policy-checked, human-confirmed process termination (shared by tools and CLI)."""
    require_psutil()
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        user = proc.username()
        cmdline = " ".join(proc.cmdline())[:200]
    except psutil.NoSuchProcess:
        return f"No process with PID {pid}."
    except psutil.AccessDenied:
        raise sandbox.violation("terminate", f"PID {pid}", f"PID {pid} belongs to another user; not allowed.")

    if pid <= 1 or pid in _own_lineage():
        raise sandbox.violation("terminate", f"PID {pid} ({name})",
                                f"PID {pid} ({name}) is part of the system or of LightX itself.")
    if sandbox.policy.is_protected_process(name):
        raise sandbox.violation("terminate", f"PID {pid} ({name})",
                                f"'{name}' is a protected system process and cannot be terminated.")
    try:
        me = psutil.Process().username()
    except psutil.Error:
        me = None
    if me and user != me:
        raise sandbox.violation("terminate", f"PID {pid} ({name})",
                                f"PID {pid} ({name}) is owned by '{user}', not you.")

    sandbox.authorize(
        "kill" if force else "terminate", f"PID {pid} ({name})", Risk.HIGH, cmdline,
    )
    try:
        proc.kill() if force else proc.terminate()
        proc.wait(timeout=5)
        return f"Process {pid} ({name}) {'killed' if force else 'terminated'}."
    except psutil.TimeoutExpired:
        return (f"Process {pid} ({name}) did not exit within 5s. "
                f"Ask the user whether to force-kill it.")
    except psutil.NoSuchProcess:
        return f"Process {pid} ({name}) has exited."


def make_task_tools(sandbox: Sandbox) -> list[BaseTool]:
    """Build task-manager tools bound to a sandbox."""
    require_psutil()

    @tool
    def system_overview() -> str:
        """Show CPU, memory, swap, disk usage, uptime, and battery."""
        return system_overview_text()

    @tool
    def list_processes(sort_by: str = "cpu", limit: int = 15, name_filter: str = "") -> str:
        """List running processes like a task manager.

        Args:
            sort_by: One of 'cpu', 'memory', 'pid', 'name'.
            limit: Number of processes to show.
            name_filter: Only show processes whose name contains this text.
        """
        rows = snapshot_processes(sort_by, min(limit, 100), name_filter)
        return format_processes(rows) if rows else "No matching processes."

    @tool
    def process_details(pid: int) -> str:
        """Show details of one process: command line, parent, threads, open files, connections.

        Args:
            pid: Process ID.
        """
        try:
            p = psutil.Process(pid)
            with p.oneshot():
                info = [
                    f"pid: {pid}  name: {p.name()}  status: {p.status()}",
                    f"user: {p.username()}  started: {datetime.fromtimestamp(p.create_time()):%Y-%m-%d %H:%M:%S}",
                    f"parent: {p.ppid()}  threads: {p.num_threads()}",
                    f"memory: {p.memory_info().rss / 1024**2:.0f}MB  cpu_times: {p.cpu_times().user:.1f}s user",
                ]
            try:
                info.append(f"exe: {p.exe()}")
                info.append(f"cmdline: {' '.join(p.cmdline())[:500]}")
                files = [f.path for f in p.open_files()][:10]
                if files:
                    info.append("open files: " + ", ".join(files))
                conns = p.net_connections(kind="inet")[:10]
                for c in conns:
                    remote = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "-"
                    info.append(f"conn: {c.laddr.ip}:{c.laddr.port} -> {remote} {c.status}")
            except psutil.AccessDenied:
                info.append("(more details need elevated privileges)")
            return "\n".join(info)
        except psutil.NoSuchProcess:
            return f"No process with PID {pid}."

    @tool
    def terminate_process(pid: int, force: bool = False) -> str:
        """Terminate a process (asks the user for confirmation first).

        Args:
            pid: Process ID to stop.
            force: Use SIGKILL instead of a graceful SIGTERM.
        """
        return terminate_pid(sandbox, pid, force)

    return [system_overview, list_processes, process_details, terminate_process]
