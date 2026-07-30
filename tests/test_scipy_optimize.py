"""Tests for scipy.optimize module."""

from unittest import mock

import zero_jax.numpy as jnp
import zero_jax.scipy.optimize as mod


def test_module_exists() -> None:
    """Test module imports correctly."""
    assert mod is not None


def test_OptimizeResults() -> None:
    """Test OptimizeResults named tuple."""
    res = mod.OptimizeResults(
        x=1,
        success=True,
        status=0,
        fun=2.0,
        jac=0.0,
        hess_inv=1.0,
        nfev=5,
        njev=5,
        nit=4,
    )
    assert res.x == 1
    assert res.success is True
    assert res.status == 0
    assert res.fun == 2.0
    assert res.jac == 0.0
    assert res.hess_inv == 1.0
    assert res.nfev == 5
    assert res.njev == 5
    assert res.nit == 4


@mock.patch("zero_jax.scipy.optimize.get_op")
def test_minimize(mock_get_op: mock.MagicMock) -> None:
    """Test minimize function routes correctly."""

    mock_op_instance = mock.MagicMock()
    mock_get_op.return_value = mock_op_instance

    # Mock return values for the compiler Op (tuple of 9 items)
    mock_op_instance.return_value = (
        jnp.array([1.0]),  # x
        jnp.array(True),  # success
        jnp.array(0),  # status
        jnp.array(2.0),  # fun
        jnp.array(0.0),  # jac
        jnp.array(1.0),  # hess_inv
        jnp.array(5),  # nfev
        jnp.array(5),  # njev
        jnp.array(4),  # nit
    )

    def dummy_fun(x: jnp.ndarray) -> jnp.ndarray:
        return x**2

    x0 = jnp.array([0.5])
    args = (1,)

    res = mod.minimize(
        fun=dummy_fun,
        x0=x0,
        args=args,
        method="BFGS",
        tol=1e-5,
        options={"maxiter": 10},
    )

    # Verify get_op was called for the correct Op
    mock_get_op.assert_called_once_with("ScipyOptimizeMinimize")

    # Verify the op was executed
    assert mock_op_instance.call_count == 1
    call_args, call_kwargs = mock_op_instance.call_args

    assert call_args[0] is dummy_fun
    assert jnp.allclose(call_args[1], x0)
    assert call_args[2] == args
    assert call_args[3] == "BFGS"
    assert call_args[4] == 1e-5
    assert call_args[5] == {"maxiter": 10}

    # Verify results wrapped properly
    assert isinstance(res, mod.OptimizeResults)
    assert jnp.allclose(res.x, jnp.array([1.0]))
    assert res.success == jnp.array(True)
    assert res.status == jnp.array(0)
    assert jnp.allclose(res.fun, jnp.array(2.0))
    assert jnp.allclose(res.jac, jnp.array(0.0))
    assert jnp.allclose(res.hess_inv, jnp.array(1.0))
    assert res.nfev == jnp.array(5)
    assert res.njev == jnp.array(5)
    assert res.nit == jnp.array(4)
