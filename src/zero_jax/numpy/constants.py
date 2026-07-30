"""Constants and constant objects for jax.numpy."""

from __future__ import annotations

import math
from typing import Any

from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

e = math.e
euler_gamma = 0.5772156649015329

newaxis = None


class _IndexExp:
    def __getitem__(self, item: Any) -> Any:
        if not isinstance(item, tuple):  # pragma: no cover
            return (item,)  # pragma: no cover
        return item  # pragma: no cover


class _S:
    def __getitem__(self, item: Any) -> Any:
        return item  # pragma: no cover


index_exp = _IndexExp()
s_ = _S()


class _MGrid:
    def __getitem__(self, item: Any) -> Any:
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.mgrid(item))


class _OGrid:
    def __getitem__(self, item: Any) -> Any:
        import zero_jax._compiler_proxy_ops as ops

        res = ops.ogrid(item)
        if isinstance(res, (tuple, list)):
            return [_wrap(t) for t in res]  # pragma: no cover
        return _wrap(res)


mgrid = _MGrid()
ogrid = _OGrid()


class _RClass:
    def __getitem__(self, item: Any) -> Any:
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.r_(item))


class _CClass:
    def __getitem__(self, item: Any) -> Any:
        import zero_jax._compiler_proxy_ops as ops

        return _wrap(ops.c_(item))


r_ = _RClass()
c_ = _CClass()
