"""AST parser CLI entry point to ensure no disallowed 3rd-party imports."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from scripts.run_checks import main as _run_checks_main


def main(argv: Sequence[str] | None = None) -> None:
    """Delegate allowed imports check to ``scripts.run_checks``.

    Args:
        argv: Optional list of file paths to check.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    _run_checks_main(["check_allowed_imports", *args])


if __name__ == "__main__":
    main()
