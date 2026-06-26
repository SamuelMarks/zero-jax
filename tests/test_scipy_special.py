"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.special as mod


def test_bernoulli() -> None:
    """Test bernoulli."""
    with patch("ml_switcheroo_compiler.ops.bernoulli") as mock_op:
        mod.bernoulli()
        mock_op.assert_called_once_with()


def test_bessel_jn() -> None:
    """Test bessel_jn."""
    with patch("ml_switcheroo_compiler.ops.bessel_jn") as mock_op:
        mod.bessel_jn()
        mock_op.assert_called_once_with()


def test_beta() -> None:
    """Test beta."""
    with patch("ml_switcheroo_compiler.ops.beta") as mock_op:
        mod.beta()
        mock_op.assert_called_once_with()


def test_betainc() -> None:
    """Test betainc."""
    with patch("ml_switcheroo_compiler.ops.betainc") as mock_op:
        mod.betainc()
        mock_op.assert_called_once_with()


def test_betaln() -> None:
    """Test betaln."""
    with patch("ml_switcheroo_compiler.ops.betaln") as mock_op:
        mod.betaln()
        mock_op.assert_called_once_with()


def test_digamma() -> None:
    """Test digamma."""
    with patch("ml_switcheroo_compiler.ops.digamma") as mock_op:
        mod.digamma()
        mock_op.assert_called_once_with()


def test_entr() -> None:
    """Test entr."""
    with patch("ml_switcheroo_compiler.ops.entr") as mock_op:
        mod.entr()
        mock_op.assert_called_once_with()


def test_erf() -> None:
    """Test erf."""
    with patch("ml_switcheroo_compiler.ops.erf") as mock_op:
        mod.erf()
        mock_op.assert_called_once_with()


def test_erfc() -> None:
    """Test erfc."""
    with patch("ml_switcheroo_compiler.ops.erfc") as mock_op:
        mod.erfc()
        mock_op.assert_called_once_with()


def test_erfinv() -> None:
    """Test erfinv."""
    with patch("ml_switcheroo_compiler.ops.erfinv") as mock_op:
        mod.erfinv()
        mock_op.assert_called_once_with()


def test_exp1() -> None:
    """Test exp1."""
    with patch("ml_switcheroo_compiler.ops.exp1") as mock_op:
        mod.exp1()
        mock_op.assert_called_once_with()


def test_expi() -> None:
    """Test expi."""
    with patch("ml_switcheroo_compiler.ops.expi") as mock_op:
        mod.expi()
        mock_op.assert_called_once_with()


def test_expit() -> None:
    """Test expit."""
    with patch("ml_switcheroo_compiler.ops.expit") as mock_op:
        mod.expit()
        mock_op.assert_called_once_with()


def test_expn() -> None:
    """Test expn."""
    with patch("ml_switcheroo_compiler.ops.expn") as mock_op:
        mod.expn()
        mock_op.assert_called_once_with()


def test_factorial() -> None:
    """Test factorial."""
    with patch("ml_switcheroo_compiler.ops.factorial") as mock_op:
        mod.factorial()
        mock_op.assert_called_once_with()


def test_gamma() -> None:
    """Test gamma."""
    with patch("ml_switcheroo_compiler.ops.gamma") as mock_op:
        mod.gamma()
        mock_op.assert_called_once_with()


def test_gammainc() -> None:
    """Test gammainc."""
    with patch("ml_switcheroo_compiler.ops.gammainc") as mock_op:
        mod.gammainc()
        mock_op.assert_called_once_with()


def test_gammaincc() -> None:
    """Test gammaincc."""
    with patch("ml_switcheroo_compiler.ops.gammaincc") as mock_op:
        mod.gammaincc()
        mock_op.assert_called_once_with()


def test_gammaln() -> None:
    """Test gammaln."""
    with patch("ml_switcheroo_compiler.ops.gammaln") as mock_op:
        mod.gammaln()
        mock_op.assert_called_once_with()


def test_gammasgn() -> None:
    """Test gammasgn."""
    with patch("ml_switcheroo_compiler.ops.gammasgn") as mock_op:
        mod.gammasgn()
        mock_op.assert_called_once_with()


def test_hyp1f1() -> None:
    """Test hyp1f1."""
    with patch("ml_switcheroo_compiler.ops.hyp1f1") as mock_op:
        mod.hyp1f1()
        mock_op.assert_called_once_with()


def test_i0() -> None:
    """Test i0."""
    with patch("ml_switcheroo_compiler.ops.i0") as mock_op:
        mod.i0()
        mock_op.assert_called_once_with()


def test_i0e() -> None:
    """Test i0e."""
    with patch("ml_switcheroo_compiler.ops.i0e") as mock_op:
        mod.i0e()
        mock_op.assert_called_once_with()


def test_i1() -> None:
    """Test i1."""
    with patch("ml_switcheroo_compiler.ops.i1") as mock_op:
        mod.i1()
        mock_op.assert_called_once_with()


def test_i1e() -> None:
    """Test i1e."""
    with patch("ml_switcheroo_compiler.ops.i1e") as mock_op:
        mod.i1e()
        mock_op.assert_called_once_with()


def test_kl_div() -> None:
    """Test kl_div."""
    with patch("ml_switcheroo_compiler.ops.kl_div") as mock_op:
        mod.kl_div()
        mock_op.assert_called_once_with()


def test_log_ndtr() -> None:
    """Test log_ndtr."""
    with patch("ml_switcheroo_compiler.ops.log_ndtr") as mock_op:
        mod.log_ndtr()
        mock_op.assert_called_once_with()


def test_logit() -> None:
    """Test logit."""
    with patch("ml_switcheroo_compiler.ops.logit") as mock_op:
        mod.logit()
        mock_op.assert_called_once_with()


def test_logsumexp() -> None:
    """Test logsumexp."""
    with patch("ml_switcheroo_compiler.ops.logsumexp") as mock_op:
        mod.logsumexp()
        mock_op.assert_called_once_with()


def test_lpmn() -> None:
    """Test lpmn."""
    with patch("ml_switcheroo_compiler.ops.lpmn") as mock_op:
        mod.lpmn()
        mock_op.assert_called_once_with()


def test_lpmn_values() -> None:
    """Test lpmn_values."""
    with patch("ml_switcheroo_compiler.ops.lpmn_values") as mock_op:
        mod.lpmn_values()
        mock_op.assert_called_once_with()


def test_multigammaln() -> None:
    """Test multigammaln."""
    with patch("ml_switcheroo_compiler.ops.multigammaln") as mock_op:
        mod.multigammaln()
        mock_op.assert_called_once_with()


def test_ndtr() -> None:
    """Test ndtr."""
    with patch("ml_switcheroo_compiler.ops.ndtr") as mock_op:
        mod.ndtr()
        mock_op.assert_called_once_with()


def test_ndtri() -> None:
    """Test ndtri."""
    with patch("ml_switcheroo_compiler.ops.ndtri") as mock_op:
        mod.ndtri()
        mock_op.assert_called_once_with()


def test_poch() -> None:
    """Test poch."""
    with patch("ml_switcheroo_compiler.ops.poch") as mock_op:
        mod.poch()
        mock_op.assert_called_once_with()


def test_polygamma() -> None:
    """Test polygamma."""
    with patch("ml_switcheroo_compiler.ops.polygamma") as mock_op:
        mod.polygamma()
        mock_op.assert_called_once_with()


def test_rel_entr() -> None:
    """Test rel_entr."""
    with patch("ml_switcheroo_compiler.ops.rel_entr") as mock_op:
        mod.rel_entr()
        mock_op.assert_called_once_with()


def test_spence() -> None:
    """Test spence."""
    with patch("ml_switcheroo_compiler.ops.spence") as mock_op:
        mod.spence()
        mock_op.assert_called_once_with()


def test_sph_harm() -> None:
    """Test sph_harm."""
    with patch("ml_switcheroo_compiler.ops.sph_harm") as mock_op:
        mod.sph_harm()
        mock_op.assert_called_once_with()


def test_xlog1py() -> None:
    """Test xlog1py."""
    with patch("ml_switcheroo_compiler.ops.xlog1py") as mock_op:
        mod.xlog1py()
        mock_op.assert_called_once_with()


def test_xlogy() -> None:
    """Test xlogy."""
    with patch("ml_switcheroo_compiler.ops.xlogy") as mock_op:
        mod.xlogy()
        mock_op.assert_called_once_with()


def test_zeta() -> None:
    """Test zeta."""
    with patch("ml_switcheroo_compiler.ops.zeta") as mock_op:
        mod.zeta()
        mock_op.assert_called_once_with()
