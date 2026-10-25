"""API parity checker against JAX snapshots and ml-ecosystem-snapshots (v0.0.5)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from scripts.run_checks import get_signatures as get_signatures
from scripts.run_checks import main as _run_checks_main


def main(argv: Sequence[str] | None = None) -> None:
    """Delegate API parity check to ``scripts.run_checks``.

    Args:
        argv: Optional list of command-line flags (e.g., ``["--update"]``).
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    _run_checks_main(["check_api_parity", *args])


__all__ = ["get_signatures", "main"]

if __name__ == "__main__":
    main()
