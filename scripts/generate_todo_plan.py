"""Utility to generate a summary plan of compiler bindings and audit TODO_PLAN.md."""

from __future__ import annotations

from ml_switcheroo_compiler import ops

import zero_jax._compiler_proxy_ops as proxy_ops


def count_todo_checkboxes(markdown_text: str) -> tuple[int, int]:
    """Count checked and unchecked markdown checkbox items in ``markdown_text``.

    Args:
        markdown_text: Markdown content to inspect.

    Returns:
        tuple[int, int]: Tuple of ``(checked_count, unchecked_count)``.
    """
    checked = 0
    unchecked = 0
    for line in markdown_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("- [x]", "- [X]")):
            checked += 1
        elif stripped.startswith("- [ ]"):
            unchecked += 1
    return checked, unchecked


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


__all__ = ["count_todo_checkboxes", "generate_plan"]
