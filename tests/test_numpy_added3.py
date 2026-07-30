from unittest.mock import MagicMock, patch

import pytest

import zero_jax.numpy as jnp
from zero_jax.lax import missing_funcs


def test_section_4_coverage():
    funcs = [
        "packbits",
        "unpackbits",
        "percentile",
        "quantile",
        "piecewise",
        "promote_types",
        "trapezoid",
        "tri",
        "tril_indices",
        "tril_indices_from",
        "triu_indices",
        "triu_indices_from",
        "trim_zeros",
        "union1d",
        "unique",
        "unwrap",
        "vander",
        "nonzero",
        "mgrid",
        "ogrid",
        "r_",
        "c_",
        "poly",
        "polyadd",
        "polyder",
        "polydiv",
        "polyfit",
        "polyint",
        "polymul",
        "polysub",
        "polyval",
        "roots",
        "vectorize",
        "broadcasted_iota",
    ]
    for func in funcs:
        patcher = patch(f"zero_jax._compiler_proxy_ops.{func}", create=True)
        m = patcher.start()
        if func == "polydiv":
            m.return_value = (1.0, 1.0)
        else:
            m.return_value = 1.0

        # for indexers
        if func in ("mgrid", "ogrid", "r_", "c_"):
            m.__getitem__ = MagicMock(return_value=1.0)

    missing_funcs.broadcasted_iota(float, (2,), 0)

    jnp.packbits([1])
    jnp.unpackbits([1])
    jnp.percentile([1], 50)
    jnp.quantile([1], 0.5)
    jnp.piecewise([1], [True], [1])
    jnp.promote_types(int, float)
    jnp.trapezoid([1])
    jnp.tri(3)
    jnp.tril_indices(3)
    jnp.tril_indices_from([[1]])
    jnp.triu_indices(3)
    jnp.triu_indices_from([[1]])
    jnp.trim_zeros([1])
    jnp.union1d([1], [1])
    jnp.unique([1])
    jnp.unwrap([1])
    jnp.vander([1])
    jnp.nonzero([1])

    _ = jnp.mgrid[0:1]
    _ = jnp.ogrid[0:1]
    _ = jnp.r_[0:1]
    _ = jnp.c_[0:1]

    jnp.poly([1])
    jnp.polyadd([1], [1])
    jnp.polyder([1])
    jnp.polydiv([1], [1])
    jnp.polyfit([1], [1], 1)
    jnp.polyint([1])
    jnp.polymul([1], [1])
    jnp.polysub([1], [1])
    jnp.polyval([1], 1)
    jnp.roots([1])

    patch.stopall()
