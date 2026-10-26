"""Tests for zero_jax.scipy.special."""

from __future__ import annotations

import math

import zero_jax.numpy as jnp
import zero_jax.scipy.special as mod
from tests.conftest import assert_allclose


def test_bernoulli() -> None:
    """Test bernoulli."""
    assert_allclose(mod.bernoulli(4), [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0])


def test_bessel_jn() -> None:
    """Test bessel_jn."""
    res = mod.bessel_jn(jnp.array([0.0]), v=1)
    assert res.shape == (2, 1)
    assert_allclose(res[0], [1.0], atol=1e-4)


def test_beta() -> None:
    """Test beta."""
    assert_allclose(mod.beta(1.0, 1.0), 1.0)
    assert_allclose(mod.beta(jnp.array([2, 3]), jnp.array([2, 3])), [1.0 / 6.0, 1.0 / 30.0], atol=1e-5)


def test_betainc() -> None:
    """Test betainc."""
    assert_allclose(mod.betainc(1.0, 1.0, 0.5), 0.5, atol=1e-5)


def test_betaln() -> None:
    """Test betaln."""
    assert_allclose(mod.betaln(1.0, 1.0), 0.0, atol=1e-5)


def test_digamma() -> None:
    """Test digamma."""
    assert_allclose(mod.digamma(1.0), -0.57721566, atol=1e-4)


def test_entr() -> None:
    """Test entr."""
    res = mod.entr(jnp.array([0.0, 1.0, math.e]))
    assert_allclose(res, [0.0, 0.0, -math.e], atol=1e-5)


def test_erf() -> None:
    """Test erf."""
    assert_allclose(mod.erf(0.0), 0.0)


def test_erfc() -> None:
    """Test erfc."""
    assert_allclose(mod.erfc(0.0), 1.0)


def test_erfinv() -> None:
    """Test erfinv."""
    assert_allclose(mod.erfinv(0.0), 0.0)


def test_exp1() -> None:
    """Test exp1."""
    res = mod.exp1(1.0)
    assert_allclose(res, -0.5772156649 + 1.0 - 0.25, atol=1e-5)


def test_expi() -> None:
    """Test expi."""
    res = mod.expi(1.0)
    assert_allclose(res, 0.5772156649 + 1.0 + 0.25, atol=1e-5)


def test_expit() -> None:
    """Test expit."""
    assert_allclose(mod.expit(0.0), 0.5)


def test_expn() -> None:
    """Test expn."""
    assert_allclose(mod.expn(1.0, 1.0), math.exp(-1.0))


def test_factorial() -> None:
    """Test factorial."""
    assert_allclose(mod.factorial(jnp.array([0.0, 1.0, 4.0])), [1.0, 1.0, 24.0], atol=1e-5)


def test_gamma() -> None:
    """Test gamma."""
    assert_allclose(mod.gamma(jnp.array([1.0, 4.0])), [1.0, 6.0], atol=1e-5)


def test_gammainc() -> None:
    """Test gammainc."""
    assert_allclose(mod.gammainc(1.0, 1.0), 1.0 - math.exp(-1.0), atol=1e-4)


def test_gammaincc() -> None:
    """Test gammaincc."""
    assert_allclose(mod.gammaincc(1.0, 1.0), math.exp(-1.0), atol=1e-4)


def test_gammaln() -> None:
    """Test gammaln."""
    assert_allclose(mod.gammaln(1.0), 0.0, atol=1e-5)


def test_gammasgn() -> None:
    """Test gammasgn."""
    assert_allclose(mod.gammasgn(jnp.array([-1.5, -0.5, 1.5])), [1.0, -1.0, 1.0])


def test_hyp1f1() -> None:
    """Test hyp1f1."""
    assert_allclose(mod.hyp1f1(1.0, 1.0, 1.0), math.e, atol=1e-4)


def test_i0() -> None:
    """Test i0."""
    assert_allclose(mod.i0(0.0), 1.0, atol=1e-5)


def test_i0e() -> None:
    """Test i0e."""
    assert_allclose(mod.i0e(0.0), 1.0, atol=1e-5)


def test_i1() -> None:
    """Test i1."""
    assert_allclose(mod.i1(0.0), 0.0, atol=1e-5)


def test_i1e() -> None:
    """Test i1e."""
    assert_allclose(mod.i1e(0.0), 0.0, atol=1e-5)


def test_kl_div() -> None:
    """Test kl_div."""
    assert_allclose(mod.kl_div(1.0, 1.0), 0.0, atol=1e-5)


def test_log_ndtr() -> None:
    """Test log_ndtr."""
    assert_allclose(mod.log_ndtr(0.0), math.log(0.5), atol=1e-5)


def test_log_softmax() -> None:
    """Test log_softmax."""
    assert_allclose(mod.log_softmax(jnp.array([0.0, 0.0])), [-math.log(2.0), -math.log(2.0)], atol=1e-5)


def test_logit() -> None:
    """Test logit."""
    assert_allclose(mod.logit(0.5), 0.0, atol=1e-5)


def test_logsumexp() -> None:
    """Test logsumexp."""
    assert_allclose(mod.logsumexp(jnp.array([0.0, 0.0])), math.log(2.0), atol=1e-5)


def test_lpmn() -> None:
    """Test lpmn."""
    p, pd = mod.lpmn(1, 2, jnp.array([0.0, 1.0]))
    assert_allclose(p, [0.0, 0.0])
    assert_allclose(pd, [0.0, 0.0])


def test_lpmn_values() -> None:
    """Test lpmn_values."""
    assert_allclose(mod.lpmn_values(1, 2, jnp.array([0.0, 1.0]), True), [1.0, 1.0])


def test_multigammaln() -> None:
    """Test multigammaln."""
    assert_allclose(mod.multigammaln(1.0, 1), 0.0, atol=1e-5)


def test_ndtr() -> None:
    """Test ndtr."""
    assert_allclose(mod.ndtr(0.0), 0.5, atol=1e-5)


def test_ndtri() -> None:
    """Test ndtri."""
    assert_allclose(mod.ndtri(0.5), 0.0, atol=1e-5)


def test_poch() -> None:
    """Test poch."""
    assert_allclose(mod.poch(2.0, 2.0), 6.0, atol=1e-5)


def test_polygamma() -> None:
    """Test polygamma."""
    assert_allclose(mod.polygamma(1, 1.0), (math.pi**2) / 6.0, atol=1e-3)


def test_rel_entr() -> None:
    """Test rel_entr."""
    assert_allclose(mod.rel_entr(0.0, 1.0), 0.0)
    assert_allclose(mod.rel_entr(2.0, 1.0), 2.0 * math.log(2.0), atol=1e-5)


def test_spence() -> None:
    """Test spence."""
    assert_allclose(mod.spence(1.0), 0.0, atol=1e-5)


def test_sph_harm() -> None:
    """Test sph_harm."""
    res = mod.sph_harm(jnp.array([0]), jnp.array([0]), jnp.array([0.0]), jnp.array([0.0]))
    assert_allclose(res, [0.5 / math.sqrt(math.pi)], atol=1e-5)


def test_xlog1py() -> None:
    """Test xlog1py."""
    assert_allclose(mod.xlog1py(jnp.array([0.0, 2.0]), jnp.array([5.0, math.e - 1.0])), [0.0, 2.0], atol=1e-5)


def test_xlogy() -> None:
    """Test xlogy."""
    assert_allclose(mod.xlogy(jnp.array([0.0, 2.0]), jnp.array([0.0, math.e])), [0.0, 2.0], atol=1e-5)


def test_zeta() -> None:
    """Test zeta."""
    assert_allclose(mod.zeta(2.0, 1.0), (math.pi**2) / 6.0, atol=1e-3)
