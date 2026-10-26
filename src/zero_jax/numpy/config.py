"""Print options configuration."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


def set_printoptions(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress: bool | None = None,
    nanstr: str | None = None,
    infstr: str | None = None,
    sign: str | None = None,
) -> None:
    """Set formatting options for array printing.

    Args:
        precision: Number of digits of precision for floating point output.
        threshold: Total number of array elements which trigger summarization.
        edgeitems: Number of array items in summary at beginning and end of each dimension.
        linewidth: Number of characters per line for the purpose of inserting line breaks.
        suppress: If ``True``, always print floating point numbers using fixed point notation.
        nanstr: String representation of floating point not-a-number.
        infstr: String representation of floating point infinity.
        sign: Controls printing of the sign of floating-point types.
    """
    _ = (precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign)


@contextlib.contextmanager
def printoptions(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress: bool | None = None,
    nanstr: str | None = None,
    infstr: str | None = None,
    sign: str | None = None,
) -> Iterator[None]:
    """Context manager for setting array print options temporarily.

    Args:
        precision: Number of digits of precision for floating point output.
        threshold: Total number of array elements which trigger summarization.
        edgeitems: Number of array items in summary at beginning and end of each dimension.
        linewidth: Number of characters per line for the purpose of inserting line breaks.
        suppress: If ``True``, always print floating point numbers using fixed point notation.
        nanstr: String representation of floating point not-a-number.
        infstr: String representation of floating point infinity.
        sign: Controls printing of the sign of floating-point types.

    Yields:
        None: Context scope with temporary print options.
    """
    _ = (precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign)
    yield
