"""Isolate tests from the developer's real ~/.lightx/config.json."""

import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    # Runs before test modules are imported, so `import lightagentx.smartos`
    # never sees a SmartOS switch the developer turned off locally.
    os.environ["LIGHTAGENTX_CONFIG"] = str(Path(tempfile.mkdtemp()) / "config.json")
    os.environ.pop("LIGHTAGENTX_SMARTOS", None)
