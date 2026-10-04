"""File tools — browse, read, write, search, open and (safely) delete files."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..sandbox import Risk, Sandbox
from ..tools.base import BaseTool, tool


_RUNNABLE_SUFFIXES = {
    ".desktop", ".sh", ".bash", ".run", ".appimage", ".jar", ".py", ".pl",
    ".exe", ".bat", ".cmd", ".ps1", ".msi", ".app", ".command", ".deb", ".rpm",
}


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def open_with_desktop(target: str) -> None:
    """Hand a file/URL to the desktop's default handler (runs outside the sandbox)."""
    if sys.platform.startswith("linux"):
        cmd = ["xdg-open", target]
    elif sys.platform == "darwin":
        cmd = ["open", target]
    elif sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    else:
        raise OSError(f"Opening files is not supported on {sys.platform}")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def make_file_tools(sandbox: Sandbox) -> list[BaseTool]:
    """Build file-management tools bound to a sandbox."""

    @tool
    def list_directory(path: str = "~", show_hidden: bool = False) -> str:
        """List the contents of a directory with sizes and modification times.

        Args:
            path: Directory to list. Relative paths are inside the agent workspace.
            show_hidden: Include dotfiles.
        """
        p = sandbox.check_read(path)
        if not p.is_dir():
            return f"Not a directory: {p}"
        rows = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if not show_hidden and child.name.startswith("."):
                continue
            if sandbox.policy.is_denied(child):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            kind = "dir " if child.is_dir() else "file"
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            size = "-" if child.is_dir() else _human_size(st.st_size)
            rows.append(f"{kind}  {size:>8}  {mtime}  {child.name}")
            if len(rows) >= 500:
                rows.append("... (truncated at 500 entries)")
                break
        return f"{p}\n" + ("\n".join(rows) if rows else "(empty)")

    @tool
    def read_file(path: str, max_chars: int = 20000) -> str:
        """Read a text file.

        Args:
            path: File to read.
            max_chars: Maximum characters to return.
        """
        p = sandbox.check_read(path)
        if not p.is_file():
            return f"Not a file: {p}"
        data = p.read_bytes()[: max(1, min(max_chars, sandbox.policy.max_output_chars)) * 4]
        if b"\x00" in data[:4096]:
            return f"{p} looks like a binary file ({_human_size(p.stat().st_size)}); not shown."
        text = data.decode(errors="replace")
        return sandbox.truncate(text[:max_chars])

    @tool
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Create or overwrite a text file (or append to it).

        Args:
            path: File to write. Relative paths go into the agent workspace.
            content: Text to write.
            append: Append instead of overwriting.
        """
        p = sandbox.check_write(path)
        risk = Risk.LOW if sandbox.in_workspace(p) else Risk.MEDIUM
        if p.exists() and not append and not sandbox.in_workspace(p):
            risk = Risk.HIGH
        sandbox.authorize("write_file", str(p), risk,
                          f"{'append' if append else 'overwrite' if p.exists() else 'create'}, "
                          f"{len(content)} chars")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {p}"

    @tool
    def search_files(pattern: str, root: str = "~", max_results: int = 50) -> str:
        """Find files whose name matches a glob pattern (e.g. '*.pdf', 'report*').

        Args:
            pattern: Glob pattern matched against file names.
            root: Directory to search under.
            max_results: Maximum number of matches to return.
        """
        base = sandbox.check_read(root)
        deadline = time.monotonic() + 15
        hits: list[str] = []
        for m in base.rglob(pattern):
            if time.monotonic() > deadline:
                hits.append("... (search stopped after 15s)")
                break
            if any(part.startswith(".") for part in m.relative_to(base).parts[:-1]):
                continue
            if sandbox.policy.is_denied(m):
                continue
            hits.append(str(m))
            if len(hits) >= max_results:
                break
        return "\n".join(hits) if hits else f"No files matching '{pattern}' under {base}"

    @tool
    def file_info(path: str) -> str:
        """Show size, owner permissions, and timestamps for a file or directory.

        Args:
            path: File or directory to inspect.
        """
        p = sandbox.check_read(path)
        st = p.stat()
        return (
            f"path: {p}\ntype: {'directory' if p.is_dir() else 'file'}\n"
            f"size: {_human_size(st.st_size)}\npermissions: {stat.filemode(st.st_mode)}\n"
            f"modified: {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}\n"
            f"created/changed: {datetime.fromtimestamp(st.st_ctime):%Y-%m-%d %H:%M:%S}"
        )

    @tool
    def open_file(path: str) -> str:
        """Open a file or folder in the user's default desktop application.

        Args:
            path: File or folder to open.
        """
        p = sandbox.check_read(path)
        if not p.exists():
            return f"Does not exist: {p}"
        # Desktop handlers may *execute* launchers and scripts — treat those as HIGH.
        runnable = p.is_file() and (
            os.access(p, os.X_OK) or p.suffix.lower() in _RUNNABLE_SUFFIXES
        )
        sandbox.authorize("open_file", str(p), Risk.HIGH if runnable else Risk.MEDIUM)
        open_with_desktop(str(p))
        return f"Opened {p} with the default application."

    @tool
    def move_to_trash(path: str) -> str:
        """Delete a file or folder by moving it to the LightX trash (recoverable).

        Args:
            path: File or folder to delete.
        """
        p = sandbox.check_write(path)
        if not p.exists():
            return f"Does not exist: {p}"
        if p in sandbox.policy.write_paths:
            raise sandbox.violation("delete", str(p), "Refusing to delete a sandbox root directory.")
        sandbox.authorize("delete", str(p), Risk.HIGH, "moved to ~/.lightx/trash (recoverable)")
        trash = Path.home() / ".lightx" / "trash" / datetime.now().strftime("%Y%m%d-%H%M%S")
        trash.mkdir(parents=True, exist_ok=True)
        dest = trash / p.name
        shutil.move(str(p), str(dest))
        return f"Moved {p} to trash at {dest}"

    return [list_directory, read_file, write_file, search_files, file_info,
            open_file, move_to_trash]
