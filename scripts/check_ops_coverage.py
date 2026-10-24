"""Operations coverage checker for zero-jax."""

from __future__ import annotations

from scripts.run_checks import main as _run_checks_main


def check_coverage() -> None:
    """Delegate operations coverage check to ``scripts.run_checks``."""
    _run_checks_main(["check_ops_coverage"])


__all__ = ["check_coverage"]

if __name__ == "__main__":
    check_coverage()
