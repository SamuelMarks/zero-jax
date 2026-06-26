"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.special as mod


def test_bernoulli() -> None:
    """Test bernoulli."""
    with pytest.raises(NotImplementedError):
        mod.bernoulli()


def test_bessel_jn() -> None:
    """Test bessel_jn."""
    with pytest.raises(NotImplementedError):
        mod.bessel_jn()


def test_beta() -> None:
    """Test beta."""
    with pytest.raises(NotImplementedError):
        mod.beta()


def test_betainc() -> None:
    """Test betainc."""
    with pytest.raises(NotImplementedError):
        mod.betainc()


def test_betaln() -> None:
    """Test betaln."""
    with pytest.raises(NotImplementedError):
        mod.betaln()


def test_digamma() -> None:
    """Test digamma."""
    with pytest.raises(NotImplementedError):
        mod.digamma()


def test_entr() -> None:
    """Test entr."""
    with pytest.raises(NotImplementedError):
        mod.entr()


def test_erf() -> None:
    """Test erf."""
    with pytest.raises(NotImplementedError):
        mod.erf()


def test_erfc() -> None:
    """Test erfc."""
    with pytest.raises(NotImplementedError):
        mod.erfc()


def test_erfinv() -> None:
    """Test erfinv."""
    with pytest.raises(NotImplementedError):
        mod.erfinv()


def test_exp1() -> None:
    """Test exp1."""
    with pytest.raises(NotImplementedError):
        mod.exp1()


def test_expi() -> None:
    """Test expi."""
    with pytest.raises(NotImplementedError):
        mod.expi()


def test_expit() -> None:
    """Test expit."""
    with pytest.raises(NotImplementedError):
        mod.expit()


def test_expn() -> None:
    """Test expn."""
    with pytest.raises(NotImplementedError):
        mod.expn()


def test_factorial() -> None:
    """Test factorial."""
    with pytest.raises(NotImplementedError):
        mod.factorial()


def test_gamma() -> None:
    """Test gamma."""
    with pytest.raises(NotImplementedError):
        mod.gamma()


def test_gammainc() -> None:
    """Test gammainc."""
    with pytest.raises(NotImplementedError):
        mod.gammainc()


def test_gammaincc() -> None:
    """Test gammaincc."""
    with pytest.raises(NotImplementedError):
        mod.gammaincc()


def test_gammaln() -> None:
    """Test gammaln."""
    with pytest.raises(NotImplementedError):
        mod.gammaln()


def test_gammasgn() -> None:
    """Test gammasgn."""
    with pytest.raises(NotImplementedError):
        mod.gammasgn()


def test_hyp1f1() -> None:
    """Test hyp1f1."""
    with pytest.raises(NotImplementedError):
        mod.hyp1f1()


def test_i0() -> None:
    """Test i0."""
    with pytest.raises(NotImplementedError):
        mod.i0()


def test_i0e() -> None:
    """Test i0e."""
    with pytest.raises(NotImplementedError):
        mod.i0e()


def test_i1() -> None:
    """Test i1."""
    with pytest.raises(NotImplementedError):
        mod.i1()


def test_i1e() -> None:
    """Test i1e."""
    with pytest.raises(NotImplementedError):
        mod.i1e()


def test_kl_div() -> None:
    """Test kl_div."""
    with pytest.raises(NotImplementedError):
        mod.kl_div()


def test_log_ndtr() -> None:
    """Test log_ndtr."""
    with pytest.raises(NotImplementedError):
        mod.log_ndtr()


def test_logit() -> None:
    """Test logit."""
    with pytest.raises(NotImplementedError):
        mod.logit()


def test_logsumexp() -> None:
    """Test logsumexp."""
    with pytest.raises(NotImplementedError):
        mod.logsumexp()


def test_lpmn() -> None:
    """Test lpmn."""
    with pytest.raises(NotImplementedError):
        mod.lpmn()


def test_lpmn_values() -> None:
    """Test lpmn_values."""
    with pytest.raises(NotImplementedError):
        mod.lpmn_values()


def test_multigammaln() -> None:
    """Test multigammaln."""
    with pytest.raises(NotImplementedError):
        mod.multigammaln()


def test_ndtr() -> None:
    """Test ndtr."""
    with pytest.raises(NotImplementedError):
        mod.ndtr()


def test_ndtri() -> None:
    """Test ndtri."""
    with pytest.raises(NotImplementedError):
        mod.ndtri()


def test_poch() -> None:
    """Test poch."""
    with pytest.raises(NotImplementedError):
        mod.poch()


def test_polygamma() -> None:
    """Test polygamma."""
    with pytest.raises(NotImplementedError):
        mod.polygamma()


def test_rel_entr() -> None:
    """Test rel_entr."""
    with pytest.raises(NotImplementedError):
        mod.rel_entr()


def test_spence() -> None:
    """Test spence."""
    with pytest.raises(NotImplementedError):
        mod.spence()


def test_sph_harm() -> None:
    """Test sph_harm."""
    with pytest.raises(NotImplementedError):
        mod.sph_harm()


def test_xlog1py() -> None:
    """Test xlog1py."""
    with pytest.raises(NotImplementedError):
        mod.xlog1py()


def test_xlogy() -> None:
    """Test xlogy."""
    with pytest.raises(NotImplementedError):
        mod.xlogy()


def test_zeta() -> None:
    """Test zeta."""
    with pytest.raises(NotImplementedError):
        mod.zeta()
