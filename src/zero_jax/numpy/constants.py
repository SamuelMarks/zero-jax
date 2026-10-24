"""Constants and constant objects for jax.numpy."""

from __future__ import annotations

import math

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _wrap

e = math.e
euler_gamma = 0.5772156649015329

newaxis = None


class _IndexExp:
    """Frontend class for ``_IndexExp``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        if not isinstance(item, tuple):
            return (item,)
        return item


class _S:
    """Frontend class for ``_S``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        return item


index_exp = _IndexExp()
s_ = _S()


class _MGrid:
    """Frontend class for ``_MGrid``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.mgrid(item))


class _OGrid:
    """Frontend class for ``_OGrid``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.ogrid(item))


mgrid = _MGrid()
ogrid = _OGrid()


class _RClass:
    """Frontend class for ``_RClass``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.r_(item))


class _CClass:
    """Frontend class for ``_CClass``."""

    def __getitem__(self, item: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``__getitem__``.

        Args:
            item: Input parameter ``item``.

        Returns:
            The operation result.
        """
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.c_(item))


r_ = _RClass()
c_ = _CClass()
