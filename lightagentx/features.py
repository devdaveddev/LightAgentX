"""Feature switches — turn SmartOS on or off without uninstalling anything.

When SmartOS is disabled, LightAgentX behaves as a plain agent framework:
``lightagentx.smartos`` refuses to import, ``SmartOS(...)`` refuses to start,
and the ``lightx-os`` command only prints how to re-enable it. All core
modules (LLMs, memory, tools, agents, hooks, snapshots) are unaffected.

The setting is resolved in this order:

    1. LIGHTAGENTX_SMARTOS environment variable  (1/0, on/off, true/false)
    2. "smartos" in ~/.lightx/config.json         (or $LIGHTAGENTX_CONFIG)
    3. default: enabled

Usage::

    import lightagentx
    lightagentx.disable_smartos()     # persists to ~/.lightx/config.json
    lightagentx.smartos_enabled()     # False
    lightagentx.enable_smartos()

Or from the terminal::

    lightx-os --disable | --enable | --status
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SMARTOS_ENV = "LIGHTAGENTX_SMARTOS"
CONFIG_ENV = "LIGHTAGENTX_CONFIG"

_TRUE = {"1", "true", "on", "yes", "enabled"}
_FALSE = {"0", "false", "off", "no", "disabled"}


class SmartOSDisabledError(ImportError):
    """Raised when SmartOS is used while it is switched off."""


def config_path() -> Path:
    """Where feature settings are stored."""
    custom = os.environ.get(CONFIG_ENV)
    return Path(custom).expanduser() if custom else Path.home() / ".lightx" / "config.json"


def _load() -> dict[str, Any]:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def smartos_status() -> dict[str, Any]:
    """Return whether SmartOS is enabled and which setting decided it."""
    env = os.environ.get(SMARTOS_ENV, "").strip().lower()
    if env in _TRUE or env in _FALSE:
        return {"enabled": env in _TRUE, "source": f"env {SMARTOS_ENV}={env}",
                "config_file": str(config_path())}
    cfg = _load()
    if isinstance(cfg.get("smartos"), bool):
        return {"enabled": cfg["smartos"], "source": "config file",
                "config_file": str(config_path())}
    return {"enabled": True, "source": "default", "config_file": str(config_path())}


def smartos_enabled() -> bool:
    return smartos_status()["enabled"]


def _set_smartos(enabled: bool) -> str:
    cfg = _load()
    cfg["smartos"] = enabled
    _save(cfg)
    msg = f"SmartOS {'enabled' if enabled else 'disabled'} (saved to {config_path()})."
    status = smartos_status()
    if status["enabled"] != enabled:
        msg += f"\nNote: {status['source']} overrides this setting in the current environment."
    return msg


def enable_smartos() -> str:
    """Turn SmartOS on (persists across sessions)."""
    return _set_smartos(True)


def disable_smartos() -> str:
    """Turn SmartOS off — LightAgentX runs as a plain agent framework."""
    return _set_smartos(False)


def require_smartos() -> None:
    """Raise SmartOSDisabledError if SmartOS is switched off."""
    status = smartos_status()
    if not status["enabled"]:
        raise SmartOSDisabledError(
            f"SmartOS is disabled ({status['source']}); LightAgentX is running as a plain "
            f"agent framework. Re-enable with `lightx-os --enable` or "
            f"`lightagentx.enable_smartos()`."
        )


def launch_smartos(argv: list[str] | None = None) -> int:
    """Entry point for the `lightx-os` command.

    Handles --enable / --disable / --status itself, so they work even while
    SmartOS is off, then hands over to the real CLI.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--enable" in args:
        print(enable_smartos())
        return 0
    if "--disable" in args:
        print(disable_smartos())
        return 0
    if "--status" in args:
        s = smartos_status()
        print(f"SmartOS: {'enabled' if s['enabled'] else 'disabled'}  "
              f"(decided by: {s['source']}; config: {s['config_file']})")
        return 0

    try:
        require_smartos()
    except SmartOSDisabledError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("SmartOS needs psutil. Install it with:  pip install \"lightagentx[os]\"",
              file=sys.stderr)
        return 1

    from .smartos.cli import main
    return main(args)
