"""Coverage badge updater for README.md."""

from __future__ import annotations

from scripts.run_checks import format_cov, get_color, get_doc_coverage, get_test_coverage
from scripts.run_checks import main as _run_checks_main


def update_readme() -> None:
    """Delegate badge update to ``scripts.run_checks``."""
    _run_checks_main(["update_badges"])


__all__ = [
    "format_cov",
    "get_color",
    "get_doc_coverage",
    "get_test_coverage",
    "update_readme",
]

if __name__ == "__main__":
    update_readme()
