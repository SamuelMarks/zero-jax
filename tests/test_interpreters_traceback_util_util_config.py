"""Tests for zero_jax module."""

import pytest

import zero_jax.interpreters.traceback_util.util.config as mod


def test_module_exists() -> None:
    """Test module."""
    assert mod is not None


def test_config_class() -> None:
    """Test Config class."""
    config = mod.Config()
    assert config is not None


def test_transfer_guard() -> None:
    """Test transfer_guard."""
    with mod.transfer_guard("allow"):
        pass
