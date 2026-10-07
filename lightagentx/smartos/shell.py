"""Shell tools — run commands and Python code inside the sandbox."""

from __future__ import annotations

import sys
import uuid

from ..sandbox import Sandbox
from ..tools.base import BaseTool, mark_guarded, tool


def make_shell_tools(sandbox: Sandbox) -> list[BaseTool]:
    """Build command-execution tools bound to a sandbox."""

    @tool
    def run_command(command: str, timeout_seconds: int = 30) -> str:
        """Run a shell command inside the sandbox (read-only system, writable workspace, no network by default).

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum run time; capped by the sandbox policy.
        """
        result = sandbox.run_shell(command, timeout_s=timeout_seconds)
        return result.to_text(sandbox.policy.max_output_chars)

    @tool
    def run_python(code: str) -> str:
        """Run a Python snippet inside the sandbox and return what it prints. Use for calculations and data processing.

        Args:
            code: Python source code. Use print() to produce output.
        """
        script = sandbox.policy.workspace / f".run_{uuid.uuid4().hex[:8]}.py"
        sandbox.authorize("run_python", f"{len(code)} chars of code", sandbox.command_risk,
                          code[:300])
        script.write_text(code, encoding="utf-8")
        try:
            python = "python3" if sys.platform != "win32" else sys.executable
            result = sandbox.run_argv([python, "-I", script.name])
        finally:
            script.unlink(missing_ok=True)
        return result.to_text(sandbox.policy.max_output_chars)

    return mark_guarded([run_command, run_python])
