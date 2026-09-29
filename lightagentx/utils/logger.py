"""Colored console logger for tracing agent reasoning chains."""

from __future__ import annotations


class _Colors:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


_TAG_STYLES = {
    "SYSTEM": (_Colors.CYAN, "⚙️"),
    "THOUGHT": (_Colors.YELLOW, "💭"),
    "ACTION": (_Colors.MAGENTA, "🔧"),
    "OBSERVATION": (_Colors.BLUE, "👁️"),
    "RESULT": (_Colors.GREEN, "✅"),
    "ERROR": (_Colors.RED, "❌"),
    "AGENT": (_Colors.CYAN, "🤖"),
    "MEMORY": (_Colors.BLUE, "🧠"),
    "PLAN": (_Colors.YELLOW, "📋"),
}


class AgentLogger:
    """
    A simple colored logger for tracing agent execution.

    Usage:
        logger = AgentLogger(verbose=True)
        logger.thought("I should look up the weather first.")
        logger.action("weather_lookup", {"city": "London"})
        logger.observation("London: 12°C, cloudy")
        logger.result("The weather in London is 12°C and cloudy.")
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, tag: str, message: str, detail: str | None = None) -> None:
        if not self.verbose:
            return
        color, emoji = _TAG_STYLES.get(tag, (_Colors.WHITE, "•"))
        header = f"{color}{_Colors.BOLD}[{tag}]{_Colors.RESET}"
        print(f"\n{emoji} {header} {message}")
        if detail:
            for line in detail.strip().split("\n"):
                print(f"   {_Colors.DIM}{line}{_Colors.RESET}")

    def system(self, message: str, detail: str | None = None) -> None:
        self._log("SYSTEM", message, detail)

    def thought(self, message: str) -> None:
        self._log("THOUGHT", message)

    def action(self, tool_name: str, args: dict | None = None) -> None:
        args_str = str(args) if args else ""
        self._log("ACTION", f"Calling `{tool_name}`", args_str)

    def observation(self, result: str) -> None:
        self._log("OBSERVATION", "Tool returned:", result)

    def result(self, message: str) -> None:
        self._log("RESULT", message)

    def error(self, message: str, detail: str | None = None) -> None:
        self._log("ERROR", message, detail)

    def agent(self, agent_name: str, message: str) -> None:
        self._log("AGENT", f"[{agent_name}] {message}")

    def memory(self, message: str) -> None:
        self._log("MEMORY", message)

    def plan(self, message: str, detail: str | None = None) -> None:
        self._log("PLAN", message, detail)

    def separator(self) -> None:
        if self.verbose:
            print(f"\n{'─' * 60}")
