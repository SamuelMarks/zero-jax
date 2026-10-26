"""Tests for zero_jax.interpreters.traceback_util.config module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config as mod


def test_module_exists() -> None:
    """Verify zero_jax.interpreters.traceback_util.config exports config singleton."""
    assert isinstance(mod.config, mod.Config)


def test_config_class() -> None:
    """Test Config class."""
    cfg = mod.Config()
    assert isinstance(cfg, mod.Config)


def test_transfer_guard() -> None:
    """Test transfer_guard."""
    with mod.transfer_guard("allow"):
        assert mod.config.read("jax_transfer_guard") == "allow"
