"""Tests for zero_jax.api.ad."""

from __future__ import annotations

import numpy as np

import zero_jax
import zero_jax.api.ad as mod
import zero_jax.numpy as jnp
from zero_jax._types import PyTree


def test_grad_and_value_and_grad() -> None:
    """Verify grad and value_and_grad with scalar, PyTree, multiple argnums, and has_aux."""
    val, g = mod.value_and_grad(lambda x: x * x)(jnp.array(3.0))
    np.testing.assert_allclose(np.asarray(val), 9.0)
    np.testing.assert_allclose(np.asarray(g), 6.0)

    # Multi-argnums and has_aux
    def fn_aux(x: PyTree, y: PyTree) -> tuple[PyTree, str]:
        """Function returning auxiliary data."""
        return x * y + x, "aux_tag"

    (v_aux, tag), g_both = mod.value_and_grad(fn_aux, argnums=(0, 1), has_aux=True)(jnp.array(2.0), jnp.array(5.0))
    np.testing.assert_allclose(np.asarray(v_aux), 12.0)
    assert tag == "aux_tag"
    np.testing.assert_allclose(np.asarray(g_both[0]), 6.0)
    np.testing.assert_allclose(np.asarray(g_both[1]), 2.0)

    # Zero-derivative function
    g_const = mod.grad(lambda x: x * 0.0)(jnp.array(3.0))
    np.testing.assert_allclose(np.asarray(g_const), 0.0)


def test_stop_gradient_in_grad() -> None:
    """Verify lax.stop_gradient zeroes out gradients through the detached branch."""
    g = mod.grad(lambda x: zero_jax.lax.stop_gradient(x) * 2.0 + x * 3.0)(jnp.array(4.0))
    np.testing.assert_allclose(np.asarray(g), 3.0)

    g_pure_stop = mod.grad(lambda x: zero_jax.lax.stop_gradient(x) * 5.0)(jnp.array(4.0))
    np.testing.assert_allclose(np.asarray(g_pure_stop), 0.0)


def test_jvp_vjp_linearize_linear_transpose() -> None:
    """Verify jvp, vjp, linearize, and linear_transpose on scalars, vectors, and has_aux."""
    p_out, t_out = mod.jvp(lambda x, y: x * y, (jnp.array(3.0), jnp.array(4.0)), (jnp.array(1.0), jnp.array(2.0)))
    np.testing.assert_allclose(np.asarray(p_out), 12.0)
    np.testing.assert_allclose(np.asarray(t_out), 4.0 * 1.0 + 3.0 * 2.0)

    p_aux, t_aux, aux_jvp = mod.jvp(
        lambda x: (x * x, "ok"),
        (jnp.array(3.0),),
        (jnp.array(2.0),),
        has_aux=True,
    )
    np.testing.assert_allclose(np.asarray(p_aux), 9.0)
    np.testing.assert_allclose(np.asarray(t_aux), 12.0)
    assert aux_jvp == "ok"

    v_out, vjp_fn = mod.vjp(lambda x, y: x * y, jnp.array(3.0), jnp.array(4.0))
    np.testing.assert_allclose(np.asarray(v_out), 12.0)
    gx, gy = vjp_fn(jnp.array(2.0))
    np.testing.assert_allclose(np.asarray(gx), 8.0)
    np.testing.assert_allclose(np.asarray(gy), 6.0)

    v_aux2, vjp_fn_aux, aux_vjp = mod.vjp(lambda x: (x * 3.0, "vjp_aux"), jnp.array(2.0), has_aux=True)
    np.testing.assert_allclose(np.asarray(v_aux2), 6.0)
    assert aux_vjp == "vjp_aux"
    np.testing.assert_allclose(np.asarray(vjp_fn_aux(jnp.array(5.0))[0]), 15.0)

    lin_y, lin_fn = mod.linearize(lambda x: x * x, jnp.array(3.0))
    np.testing.assert_allclose(np.asarray(lin_y), 9.0)
    np.testing.assert_allclose(np.asarray(lin_fn(jnp.array(2.0))), 12.0)

    lin_y_aux, lin_fn_aux, l_aux = mod.linearize(lambda x: (x * 4.0, "l_aux"), jnp.array(2.0), has_aux=True)
    np.testing.assert_allclose(np.asarray(lin_y_aux), 8.0)
    assert l_aux == "l_aux"
    np.testing.assert_allclose(np.asarray(lin_fn_aux(jnp.array(3.0))), 12.0)

    lt_fn = mod.linear_transpose(lambda x: x * 5.0, jnp.array(1.0))
    np.testing.assert_allclose(np.asarray(lt_fn(jnp.array(2.0))[0]), 10.0)
    assert mod.linear_transpose(lambda x: x)(3.0) == 3.0


def test_jacfwd_jacrev_jacobian_hessian() -> None:
    """Verify jacfwd, jacrev, jacobian, and hessian on scalar and vector functions."""
    x_vec = jnp.array([2.0, 3.0])
    jac = mod.jacrev(lambda v: v * v)(x_vec)
    np.testing.assert_allclose(np.asarray(jac), np.diag([4.0, 6.0]))

    jac_f = mod.jacfwd(lambda v: v * v)(x_vec)
    np.testing.assert_allclose(np.asarray(jac_f), np.diag([4.0, 6.0]))

    jac_alias = mod.jacobian(lambda v: v * v)(x_vec)
    np.testing.assert_allclose(np.asarray(jac_alias), np.diag([4.0, 6.0]))

    jac_aux, tag = mod.jacrev(lambda v: (v * 2.0, "j_aux"), has_aux=True)(x_vec)
    assert tag == "j_aux"
    np.testing.assert_allclose(np.asarray(jac_aux), np.eye(2) * 2.0)

    hess = mod.hessian(lambda s: s * s * s)(jnp.array(2.0))
    np.testing.assert_allclose(np.asarray(hess), 12.0, rtol=1e-3, atol=1e-3)

    hess_aux, h_tag = mod.hessian(lambda s: (s * s, "h_aux"), has_aux=True)(jnp.array(3.0))
    assert h_tag == "h_aux"
    np.testing.assert_allclose(np.asarray(hess_aux), 2.0, rtol=1e-3, atol=1e-3)

    hess_vec = mod.hessian(lambda v: jnp.sum(v * v))(x_vec)
    np.testing.assert_allclose(np.asarray(hess_vec), np.eye(2) * 2.0, rtol=1e-3, atol=1e-3)


def test_custom_jvp_custom_vjp_custom_gradient() -> None:
    """Verify custom_jvp, custom_vjp, and custom_gradient rules execute during AD."""

    @mod.custom_jvp
    def f_jvp(x: PyTree) -> PyTree:
        """Function with custom JVP."""
        return x * x

    @f_jvp.defjvp
    def f_jvp_rule(primals: tuple[PyTree, ...], tangents: tuple[PyTree, ...]) -> tuple[PyTree, PyTree]:
        """Custom JVP rule scaling tangent by 10."""
        (x,) = primals
        (t,) = tangents
        return x * x, t * 10.0

    assert f_jvp(jnp.array(3.0)) == jnp.array(9.0)
    p_c, t_c = mod.jvp(f_jvp, (jnp.array(3.0),), (jnp.array(2.0),))
    np.testing.assert_allclose(np.asarray(p_c), 9.0)
    np.testing.assert_allclose(np.asarray(t_c), 20.0)
    np.testing.assert_allclose(np.asarray(mod.grad(f_jvp)(jnp.array(3.0))), 10.0)

    @mod.custom_vjp
    def f_vjp(x: PyTree) -> PyTree:
        """Function with custom VJP."""
        return x * 2.0

    f_vjp.defvjp(lambda x: (x * 2.0, x), lambda res, g: (g * 42.0,))
    assert f_vjp(jnp.array(3.0)) == jnp.array(6.0)
    np.testing.assert_allclose(np.asarray(mod.grad(f_vjp)(jnp.array(3.0))), 42.0)
    np.testing.assert_allclose(np.asarray(mod.grad(f_vjp, argnums=(0,))(jnp.array(3.0))[0]), 42.0)
    _, c_vjp_fn = mod.vjp(f_vjp, jnp.array(3.0))
    np.testing.assert_allclose(np.asarray(c_vjp_fn(jnp.array(2.0))[0]), 84.0)

    @mod.custom_gradient
    def f_cg(x: PyTree) -> tuple[PyTree, PyTree]:
        """Function with custom_gradient."""
        return x * 3.0, lambda g: g * 99.0

    np.testing.assert_allclose(np.asarray(f_cg(jnp.array(2.0))), 6.0)
    np.testing.assert_allclose(np.asarray(mod.grad(f_cg)(jnp.array(2.0))), 99.0)

    # Plain return fallback in custom_gradient
    f_cg_plain = mod.custom_gradient(lambda x: x * 2.0)
    np.testing.assert_allclose(np.asarray(f_cg_plain(jnp.array(4.0))), 8.0)
    _, plain_vjp = mod.vjp(f_cg_plain, jnp.array(4.0))
    np.testing.assert_allclose(np.asarray(plain_vjp(jnp.array(5.0))[0]), 5.0)
