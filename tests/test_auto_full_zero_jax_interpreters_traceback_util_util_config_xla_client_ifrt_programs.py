"""Tests for zero_jax.interpreters.traceback_util.util.config.xla_client.ifrt_programs."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.xla_client.ifrt_programs as mod


def test_ifrt_programs() -> None:
    """Verify CompileOptions, Program, and make_* helpers."""
    hlo_prog = mod.make_hlo_program(b"module {}")
    assert isinstance(hlo_prog, mod.Program)
    assert hlo_prog.kind == "hlo"
