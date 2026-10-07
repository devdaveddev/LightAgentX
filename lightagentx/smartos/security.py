"""Security tools — read-only health checks for the host OS."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

from ..sandbox import Sandbox
from ..tools.base import BaseTool, mark_guarded, tool
from .tasks import require_psutil

_SUSPICIOUS_EXE_DIRS = ("/tmp/", "/dev/shm/", "/var/tmp/", "/run/user/")


def _cmd(argv: list[str]) -> str | None:
    """Run a fixed, trusted read-only system query. Never takes LLM input."""
    if not shutil.which(argv[0]):
        return None
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def listening_ports() -> list[str]:
    import psutil
    rows = []
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return ["(listing sockets needs elevated privileges on this OS)"]
    for c in conns:
        if c.status != psutil.CONN_LISTEN:
            continue
        name = "?"
        if c.pid:
            try:
                name = psutil.Process(c.pid).name()
            except psutil.Error:
                pass
        exposed = "EXPOSED" if c.laddr.ip in ("0.0.0.0", "::") else "local"
        rows.append(f"{c.laddr.ip}:{c.laddr.port:<6} {exposed:<8} pid={c.pid or '?'} {name}")
    return sorted(set(rows))


def suspicious_processes() -> list[str]:
    import psutil
    findings = []
    for p in psutil.process_iter(["pid", "name", "exe", "username"]):
        exe = p.info.get("exe") or ""
        if not exe:
            continue
        if exe.startswith(_SUSPICIOUS_EXE_DIRS):
            findings.append(f"pid={p.pid} {p.info['name']} runs from temp dir: {exe}")
        elif exe.endswith(" (deleted)"):
            findings.append(f"pid={p.pid} {p.info['name']} binary was deleted after start: {exe}")
    return findings


def credential_permissions() -> list[str]:
    """Check metadata only (never contents) of sensitive files."""
    findings = []
    home = Path.home()
    ssh = home / ".ssh"
    if ssh.is_dir():
        mode = stat.S_IMODE(ssh.stat().st_mode)
        if mode & 0o077:
            findings.append(f"~/.ssh is {oct(mode)}; should be 0o700")
        for f in ssh.iterdir():
            if f.is_file() and not f.name.endswith(".pub") and f.name not in ("known_hosts", "config", "authorized_keys", "known_hosts.old"):
                m = stat.S_IMODE(f.stat().st_mode)
                if m & 0o077:
                    findings.append(f"~/.ssh/{f.name} is {oct(m)}; private keys should be 0o600")
    for rel in (".netrc", ".git-credentials", ".pgpass", ".aws/credentials", ".docker/config.json"):
        f = home / rel
        if f.is_file():
            m = stat.S_IMODE(f.stat().st_mode)
            if m & 0o044:
                findings.append(f"~/{rel} is readable by others ({oct(m)})")
    return findings


def firewall_status() -> str:
    if sys.platform == "darwin":
        out = _cmd(["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"])
        return out or "unknown"
    if sys.platform == "win32":
        out = _cmd(["netsh", "advfirewall", "show", "allprofiles", "state"])
        return out or "unknown"
    for argv in (["firewall-cmd", "--state"], ["ufw", "status"], ["nft", "list", "tables"]):
        out = _cmd(argv)
        if out:
            return f"{argv[0]}: {out.splitlines()[0]}"
    return "no firewall tool found (firewalld/ufw/nftables)"


def mac_status() -> str:
    out = _cmd(["getenforce"])
    if out:
        return f"SELinux: {out}"
    aa = Path("/sys/module/apparmor/parameters/enabled")
    if aa.exists():
        return f"AppArmor enabled: {aa.read_text().strip()}"
    return "no SELinux/AppArmor detected"


def autostart_entries() -> list[str]:
    """Things that start automatically — the usual persistence spots."""
    rows = []
    for d in (Path.home() / ".config/autostart", Path("/etc/xdg/autostart")):
        if d.is_dir():
            rows += [f"autostart: {f}" for f in sorted(d.glob("*.desktop"))]
    units = _cmd(["systemctl", "--user", "list-unit-files", "--state=enabled", "--no-legend"])
    if units:
        rows += [f"systemd --user: {line.split()[0]}" for line in units.splitlines() if line.strip()]
    cron = _cmd(["crontab", "-l"])
    if cron and "no crontab" not in cron.lower():
        rows += [f"crontab: {line}" for line in cron.splitlines()
                 if line.strip() and not line.startswith("#")]
    return rows


def make_security_tools(sandbox: Sandbox) -> list[BaseTool]:
    """Build security-inspection tools. All are read-only (LOW risk)."""
    require_psutil()

    @tool
    def security_scan() -> str:
        """Run a full read-only security health check: firewall, SELinux/AppArmor, exposed ports, suspicious processes, credential file permissions, autostart entries."""
        ports = listening_ports()
        procs = suspicious_processes()
        perms = credential_permissions()
        auto = autostart_entries()
        sections = [
            f"## Firewall\n{firewall_status()}",
            f"## Mandatory access control\n{mac_status()}",
            "## Listening ports\n" + ("\n".join(ports) or "none"),
            "## Suspicious processes\n" + ("\n".join(procs) or "none found"),
            "## Credential file permissions\n" + ("\n".join(perms) or "ok"),
            "## Autostart / persistence\n" + ("\n".join(auto[:40]) or "none"),
            f"## Agent sandbox\n{sandbox.describe()}",
        ]
        return sandbox.truncate("\n\n".join(sections))

    @tool
    def list_network_connections(include_listening: bool = False) -> str:
        """Show active network connections with the owning process.

        Args:
            include_listening: Also include listening sockets.
        """
        import psutil
        try:
            conns = psutil.net_connections(kind="inet")
        except psutil.AccessDenied:
            return "Listing connections needs elevated privileges on this OS."
        rows = []
        for c in conns:
            if c.status == psutil.CONN_LISTEN and not include_listening:
                continue
            if not c.raddr and c.status != psutil.CONN_LISTEN:
                continue
            name = "?"
            if c.pid:
                try:
                    name = psutil.Process(c.pid).name()
                except psutil.Error:
                    pass
            remote = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "-"
            rows.append(f"{name:<20} pid={c.pid or '?':<7} {c.laddr.ip}:{c.laddr.port} -> {remote} {c.status}")
        return "\n".join(sorted(rows)[:150]) or "No active connections."

    @tool
    def check_listening_ports() -> str:
        """List ports that accept incoming connections and flag ones exposed to the network."""
        return "\n".join(listening_ports()) or "No listening ports."

    @tool
    def check_autostart() -> str:
        """List programs configured to start automatically (autostart, systemd user units, cron)."""
        return "\n".join(autostart_entries()) or "No autostart entries found."

    return mark_guarded([security_scan, list_network_connections, check_listening_ports, check_autostart])
