"""Tests for zero_jax.interpreters.traceback_util.config.xla_client.ifrt_programs."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config.xla_client.ifrt_programs as mod


def test_ifrt_programs() -> None:
    """Verify CompileOptions, Program, and make_* helpers."""
    hlo_prog = mod.make_hlo_program("module {}")
    assert isinstance(hlo_prog, mod.Program)
    assert hlo_prog.kind == "hlo"
    plug_prog = mod.make_plugin_program(b"plugin")
    assert plug_prog.kind == "plugin"
    opts = mod.make_plugin_compile_options()
    assert isinstance(opts, mod.CompileOptions)
    assert mod.make_xla_compile_options(opts) is opts
    assert isinstance(mod.make_xla_compile_options(None), mod.CompileOptions)
