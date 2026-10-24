"""Utility to generate a summary plan of compiler bindings used by zero-jax."""

from __future__ import annotations

from ml_switcheroo_compiler import ops

import zero_jax._compiler_proxy_ops as proxy_ops


def generate_plan() -> str:
    """Return a markdown summary of compiler bindings used by ``zero-jax``.

    Returns:
        str: Markdown summary string.
    """
    compiler_ops = sorted(set(ops.__all__))
    bound_ops = sorted(set(proxy_ops.__all__) & set(compiler_ops))
    sep = chr(10)
    lines = [
        "# ml-switcheroo-compiler JAX Parity Plan",
        "",
        f"- Total compiler operations: {len(compiler_ops)}",
        f"- Bound proxy operations: {len(bound_ops)}",
    ]
    return sep.join(lines)


__all__ = ["generate_plan"]
