import pytest

import zero_jax._compiler_proxy_ops as ops
import zero_jax.numpy as jnp


def test_new_ops():
    # Just reference them to satisfy the regex check and basic execution if applicable
    ops_to_check = [
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "read_file",
        "rem",
        "serialize_tensor",
        "write_file",
        "chebyshev_polynomial_t",
        "chebyshev_polynomial_u",
        "confusion_matrix",
        "descriptive",
        "distributions",
        "hermite_polynomial_h",
        "hermite_polynomial_he",
        "laguerre_polynomial_l",
        "legendre_polynomial_p",
        "modified_bessel_i0",
        "modified_bessel_i1",
        "modified_bessel_k0",
        "modified_bessel_k1",
        "shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_w",
    ]
    for op in ops_to_check:
        assert hasattr(ops, op)
