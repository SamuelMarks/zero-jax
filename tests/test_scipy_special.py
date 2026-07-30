"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.special as mod


def test_bernoulli() -> None:
    """Test bernoulli."""
    with patch("zero_jax._compiler_proxy_ops.bernoulli", create=True) as mock_op:
        mod.bernoulli()
        mock_op.assert_called_once_with()


def test_bessel_jn() -> None:
    """Test bessel_jn."""
    with patch("zero_jax._compiler_proxy_ops.bessel_jn", create=True) as mock_op:
        mod.bessel_jn()
        mock_op.assert_called_once_with()


def test_beta() -> None:
    """Test beta."""
    with patch("zero_jax._compiler_proxy_ops.beta", create=True) as mock_op:
        mod.beta()
        mock_op.assert_called_once_with()


def test_betainc() -> None:
    """Test betainc."""
    with patch("zero_jax._compiler_proxy_ops.betainc", create=True) as mock_op:
        mod.betainc()
        mock_op.assert_called_once_with()


def test_betaln() -> None:
    """Test betaln."""
    with patch("zero_jax._compiler_proxy_ops.betaln", create=True) as mock_op:
        mod.betaln()
        mock_op.assert_called_once_with()


def test_digamma() -> None:
    """Test digamma."""
    with patch("zero_jax._compiler_proxy_ops.digamma", create=True) as mock_op:
        mod.digamma()
        mock_op.assert_called_once_with()


def test_entr() -> None:
    """Test entr."""
    with patch("zero_jax._compiler_proxy_ops.entr", create=True) as mock_op:
        mod.entr()
        mock_op.assert_called_once_with()


def test_erf() -> None:
    """Test erf."""
    with patch("zero_jax._compiler_proxy_ops.erf", create=True) as mock_op:
        mod.erf()
        mock_op.assert_called_once_with()


def test_erfc() -> None:
    """Test erfc."""
    with patch("zero_jax._compiler_proxy_ops.erfc", create=True) as mock_op:
        mod.erfc()
        mock_op.assert_called_once_with()


def test_erfinv() -> None:
    """Test erfinv."""
    with patch("zero_jax._compiler_proxy_ops.erfinv", create=True) as mock_op:
        mod.erfinv()
        mock_op.assert_called_once_with()


def test_exp1() -> None:
    """Test exp1."""
    with patch("zero_jax._compiler_proxy_ops.exp1", create=True) as mock_op:
        mod.exp1()
        mock_op.assert_called_once_with()


def test_expi() -> None:
    """Test expi."""
    with patch("zero_jax._compiler_proxy_ops.expi", create=True) as mock_op:
        mod.expi()
        mock_op.assert_called_once_with()


def test_expit() -> None:
    """Test expit."""
    with patch("zero_jax._compiler_proxy_ops.expit", create=True) as mock_op:
        mod.expit()
        mock_op.assert_called_once_with()


def test_expn() -> None:
    """Test expn."""
    with patch("zero_jax._compiler_proxy_ops.expn", create=True) as mock_op:
        mod.expn()
        mock_op.assert_called_once_with()


def test_factorial() -> None:
    """Test factorial."""
    with patch("zero_jax._compiler_proxy_ops.factorial", create=True) as mock_op:
        mod.factorial()
        mock_op.assert_called_once_with()


def test_gamma() -> None:
    """Test gamma."""
    with patch("zero_jax._compiler_proxy_ops.gamma", create=True) as mock_op:
        mod.gamma()
        mock_op.assert_called_once_with()


def test_gammainc() -> None:
    """Test gammainc."""
    with patch("zero_jax._compiler_proxy_ops.gammainc", create=True) as mock_op:
        mod.gammainc()
        mock_op.assert_called_once_with()


def test_gammaincc() -> None:
    """Test gammaincc."""
    with patch("zero_jax._compiler_proxy_ops.gammaincc", create=True) as mock_op:
        mod.gammaincc()
        mock_op.assert_called_once_with()


def test_gammaln() -> None:
    """Test gammaln."""
    with patch("zero_jax._compiler_proxy_ops.gammaln", create=True) as mock_op:
        mod.gammaln()
        mock_op.assert_called_once_with()


def test_gammasgn() -> None:
    """Test gammasgn."""
    with patch("zero_jax._compiler_proxy_ops.gammasgn", create=True) as mock_op:
        mod.gammasgn()
        mock_op.assert_called_once_with()


def test_hyp1f1() -> None:
    """Test hyp1f1."""
    with patch("zero_jax._compiler_proxy_ops.hyp1f1", create=True) as mock_op:
        mod.hyp1f1()
        mock_op.assert_called_once_with()


def test_i0() -> None:
    """Test i0."""
    with patch("zero_jax._compiler_proxy_ops.i0", create=True) as mock_op:
        mod.i0()
        mock_op.assert_called_once_with()


def test_i0e() -> None:
    """Test i0e."""
    with patch("zero_jax._compiler_proxy_ops.i0e", create=True) as mock_op:
        mod.i0e()
        mock_op.assert_called_once_with()


def test_i1() -> None:
    """Test i1."""
    with patch("zero_jax._compiler_proxy_ops.i1", create=True) as mock_op:
        mod.i1()
        mock_op.assert_called_once_with()


def test_i1e() -> None:
    """Test i1e."""
    with patch("zero_jax._compiler_proxy_ops.i1e", create=True) as mock_op:
        mod.i1e()
        mock_op.assert_called_once_with()


def test_kl_div() -> None:
    """Test kl_div."""
    with patch("zero_jax._compiler_proxy_ops.kl_div", create=True) as mock_op:
        mod.kl_div()
        mock_op.assert_called_once_with()


def test_log_ndtr() -> None:
    """Test log_ndtr."""
    with patch("zero_jax._compiler_proxy_ops.log_ndtr", create=True) as mock_op:
        mod.log_ndtr()
        mock_op.assert_called_once_with()


def test_logit() -> None:
    """Test logit."""
    with patch("zero_jax._compiler_proxy_ops.logit", create=True) as mock_op:
        mod.logit()
        mock_op.assert_called_once_with()


def test_logsumexp() -> None:
    """Test logsumexp."""
    with patch("zero_jax._compiler_proxy_ops.logsumexp", create=True) as mock_op:
        mod.logsumexp()
        mock_op.assert_called_once_with()


def test_lpmn() -> None:
    """Test lpmn."""
    with patch("zero_jax._compiler_proxy_ops.lpmn", create=True) as mock_op:
        mod.lpmn()
        mock_op.assert_called_once_with()


def test_lpmn_values() -> None:
    """Test lpmn_values."""
    with patch("zero_jax._compiler_proxy_ops.lpmn_values", create=True) as mock_op:
        mod.lpmn_values()
        mock_op.assert_called_once_with()


def test_multigammaln() -> None:
    """Test multigammaln."""
    with patch("zero_jax._compiler_proxy_ops.multigammaln", create=True) as mock_op:
        mod.multigammaln()
        mock_op.assert_called_once_with()


def test_ndtr() -> None:
    """Test ndtr."""
    with patch("zero_jax._compiler_proxy_ops.ndtr", create=True) as mock_op:
        mod.ndtr()
        mock_op.assert_called_once_with()


def test_ndtri() -> None:
    """Test ndtri."""
    with patch("zero_jax._compiler_proxy_ops.ndtri", create=True) as mock_op:
        mod.ndtri()
        mock_op.assert_called_once_with()


def test_poch() -> None:
    """Test poch."""
    with patch("zero_jax._compiler_proxy_ops.poch", create=True) as mock_op:
        mod.poch()
        mock_op.assert_called_once_with()


def test_polygamma() -> None:
    """Test polygamma."""
    with patch("zero_jax._compiler_proxy_ops.polygamma", create=True) as mock_op:
        mod.polygamma()
        mock_op.assert_called_once_with()


def test_rel_entr() -> None:
    """Test rel_entr."""
    with patch("zero_jax._compiler_proxy_ops.rel_entr", create=True) as mock_op:
        mod.rel_entr()
        mock_op.assert_called_once_with()


def test_spence() -> None:
    """Test spence."""
    with patch("zero_jax._compiler_proxy_ops.spence", create=True) as mock_op:
        mod.spence()
        mock_op.assert_called_once_with()


def test_sph_harm() -> None:
    """Test sph_harm."""
    with patch("zero_jax._compiler_proxy_ops.sph_harm", create=True) as mock_op:
        mod.sph_harm()
        mock_op.assert_called_once_with()


def test_xlog1py() -> None:
    """Test xlog1py."""
    with patch("zero_jax._compiler_proxy_ops.xlog1py", create=True) as mock_op:
        mod.xlog1py()
        mock_op.assert_called_once_with()


def test_xlogy() -> None:
    """Test xlogy."""
    with patch("zero_jax._compiler_proxy_ops.xlogy", create=True) as mock_op:
        mod.xlogy()
        mock_op.assert_called_once_with()


def test_zeta() -> None:
    """Test zeta."""
    with patch("zero_jax._compiler_proxy_ops.zeta", create=True) as mock_op:
        mod.zeta()
        mock_op.assert_called_once_with()
