"""Tests for zero_jax.interpreters.traceback_util.config.xla_client.profiler."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config.xla_client.profiler as mod


def test_xla_client_profiler() -> None:
    """Verify ProfileOptions, ProfilerServer, ProfilerSession, TraceMe, and profiler functions."""
    opts = mod.ProfileOptions(include_dataset_ops=False)
    sess = mod.ProfilerSession(opts)
    assert sess.options is opts
    assert isinstance(mod.ProfilerSession().options, mod.ProfileOptions)
    srv = mod.start_server(9012)
    assert isinstance(srv, mod.ProfilerServer)
    assert srv.port == 9012
    tm = mod.TraceMe("step")
    assert tm.name == "step"
    assert mod.aggregate_profiled_instructions([b"a", b"b"]) == b"ab"
    assert mod.get_fdo_profile(b"proto", as_textproto=False) == b"proto"
    assert mod.get_fdo_profile(b"proto", as_textproto=True) == "proto"
    assert mod.get_profiled_instructions_proto("/tmp/tb") == b"/tmp/tb"
    assert mod.register_plugin_profiler() is True
