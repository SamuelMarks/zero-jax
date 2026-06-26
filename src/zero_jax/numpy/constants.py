"""Constants and constant objects for jax.numpy."""

from __future__ import annotations

import math

np = __import__("numpy")
from typing import Any

e = math.e
euler_gamma = 0.5772156649015329

newaxis = None


class _IndexExp:
    def __getitem__(self, item: Any) -> Any:
        if not isinstance(item, tuple):
            return (item,)
        return item  # pragma: no cover


index_exp = _IndexExp()
s_ = np.s_


class _MGrid:
    def __getitem__(self, item: Any) -> Any:
        return np.mgrid[item]  # pragma: no cover


class _OGrid:
    def __getitem__(self, item: Any) -> Any:
        return np.ogrid[item]  # pragma: no cover


mgrid = _MGrid()
ogrid = _OGrid()


class _RClass:
    def __getitem__(self, item: Any) -> Any:
        from .lax_numpy import array

        return array(np.r_[item])


class _CClass:
    def __getitem__(self, item: Any) -> Any:
        from .lax_numpy import array

        return array(np.c_[item])


r_ = _RClass()
c_ = _CClass()
