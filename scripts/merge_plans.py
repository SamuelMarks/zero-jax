"""Utility to merge markdown plan documents."""

from __future__ import annotations


def merge_markdown_plans(primary: str, secondary: str) -> str:
    """Merge two markdown plan strings into a single combined document.

    Args:
        primary: Primary markdown plan text.
        secondary: Secondary markdown plan text.

    Returns:
        str: Combined markdown plan text.
    """
    sep = chr(10) * 2
    return f"{primary.rstrip()}{sep}{secondary.lstrip()}"


__all__ = ["merge_markdown_plans"]
