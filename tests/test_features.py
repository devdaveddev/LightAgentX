"""Tests for the SmartOS on/off switch."""

import json
import os
import subprocess
import sys

import pytest

import lightagentx
from lightagentx.features import (
    SMARTOS_ENV,
    SmartOSDisabledError,
    launch_smartos,
    require_smartos,
)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("LIGHTAGENTX_CONFIG", str(path))
    monkeypatch.delenv(SMARTOS_ENV, raising=False)
    return path


def run_py(code, cfg_path, **env):
    full_env = {**os.environ, "LIGHTAGENTX_CONFIG": str(cfg_path), **env}
    full_env.pop(SMARTOS_ENV, None) if SMARTOS_ENV not in env else None
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=full_env, cwd=os.path.dirname(os.path.dirname(__file__)))


class TestSwitch:
    def test_enabled_by_default(self, cfg):
        assert lightagentx.smartos_enabled()
        assert lightagentx.smartos_status()["source"] == "default"

    def test_disable_persists_and_enable_restores(self, cfg):
        lightagentx.disable_smartos()
        assert json.loads(cfg.read_text())["smartos"] is False
        assert not lightagentx.smartos_enabled()
        lightagentx.enable_smartos()
        assert lightagentx.smartos_enabled()

    def test_keeps_other_config_keys(self, cfg):
        cfg.write_text(json.dumps({"theme": "dark"}))
        lightagentx.disable_smartos()
        assert json.loads(cfg.read_text()) == {"theme": "dark", "smartos": False}

    def test_env_overrides_config(self, cfg, monkeypatch):
        lightagentx.disable_smartos()
        monkeypatch.setenv(SMARTOS_ENV, "1")
        assert lightagentx.smartos_enabled()
        monkeypatch.setenv(SMARTOS_ENV, "off")
        assert "overrides" in lightagentx.enable_smartos()
        assert not lightagentx.smartos_enabled()

    def test_corrupt_config_falls_back_to_default(self, cfg):
        cfg.write_text("{not json")
        assert lightagentx.smartos_enabled()

    def test_require_raises_when_disabled(self, cfg):
        lightagentx.disable_smartos()
        with pytest.raises(SmartOSDisabledError, match="plain agent framework"):
            require_smartos()

    def test_agent_config_file_is_off_limits(self, cfg):
        from pathlib import Path
        from lightagentx import SandboxPolicy
        assert not SandboxPolicy().can_read(Path.home() / ".lightx" / "config.json")


class TestDisabledBehaviour:
    def test_core_framework_still_works(self, cfg):
        lightagentx.disable_smartos()
        r = run_py(
            "import lightagentx\n"
            "from lightagentx import SingleAgent, AgentLoop, tool, BufferMemory, HookRegistry\n"
            "print('core ok', lightagentx.smartos_enabled())", cfg)
        assert r.returncode == 0, r.stderr
        assert "core ok False" in r.stdout

    def test_smartos_import_blocked(self, cfg):
        lightagentx.disable_smartos()
        r = run_py(
            "try:\n import lightagentx.smartos\nexcept ImportError as e:\n"
            " print(type(e).__name__, e)", cfg)
        assert "SmartOSDisabledError" in r.stdout

    def test_runtime_disable_blocks_new_smartos(self, cfg):
        pytest.importorskip("psutil")
        from lightagentx.smartos import SmartOS
        lightagentx.disable_smartos()
        with pytest.raises(SmartOSDisabledError):
            SmartOS(llm=None)  # switch is checked before the LLM is touched

    def test_launcher(self, cfg, capsys):
        assert launch_smartos(["--disable"]) == 0
        assert launch_smartos([]) == 1
        assert "disabled" in capsys.readouterr().err
        assert launch_smartos(["--enable"]) == 0
        launch_smartos(["--status"])
        assert "SmartOS: enabled" in capsys.readouterr().out
