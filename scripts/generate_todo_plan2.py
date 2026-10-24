"""Secondary utility to generate a summary plan of compiler bindings."""

from __future__ import annotations

import zero_jax._compiler_proxy_creation as proxy_creation
import zero_jax._compiler_proxy_ops as proxy_ops


def generate_plan2() -> str:
    """Return a secondary markdown summary of compiler bindings.

    Returns:
        str: Markdown summary string.
    """
    creation_ops = sorted(name for name in proxy_creation.__all__ if not name.startswith("_"))
    proxy_exports = sorted(name for name in proxy_ops.__all__ if not name.startswith("_"))
    sep = chr(10)
    lines = [
        "# ml-switcheroo-compiler JAX Parity Plan (Extended)",
        "",
        f"- Creation proxy operations: {len(creation_ops)}",
        f"- Total proxy exports: {len(proxy_exports)}",
    ]
    return sep.join(lines)


__all__ = ["generate_plan2"]
