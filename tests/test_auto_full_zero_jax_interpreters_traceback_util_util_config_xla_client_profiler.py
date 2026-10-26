"""Tests for zero_jax.interpreters.traceback_util.util.config.xla_client.profiler."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.xla_client.profiler as mod


def test_xla_client_profiler() -> None:
    """Verify profiler re-exports."""
    srv = mod.start_server(9012)
    assert isinstance(srv, mod.ProfilerServer)
