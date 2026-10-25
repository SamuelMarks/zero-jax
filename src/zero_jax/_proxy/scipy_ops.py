"""SciPy special, FFT, signal, integration, and spatial proxy operations."""

from __future__ import annotations

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.ops.base import OpDef, get_op

bessel_i0e = _original_ops.bessel_i0e
bessel_i1e = _original_ops.bessel_i1e
betainc = _original_ops.betainc
chebyshev_polynomial_t = _original_ops.chebyshev_polynomial_t
chebyshev_polynomial_u = _original_ops.chebyshev_polynomial_u
convolve = _original_ops.convolve
convolve2d = _original_ops.convolve2d
correlate = _original_ops.correlate
digamma = _original_ops.digamma
erf = _original_ops.erf
erf_inv = _original_ops.erf_inv
erfc = _original_ops.erfc
erfinv = _original_ops.erfinv
fftconvolve = _original_ops.fftconvolve
fftn = _original_ops.fftn
fftshift = _original_ops.fftshift
hermite_polynomial_h = _original_ops.hermite_polynomial_h
hermite_polynomial_he = _original_ops.hermite_polynomial_he
hfft = _original_ops.hfft
ifft = _original_ops.ifft
ifft2 = _original_ops.ifft2
ifftn = _original_ops.ifftn
ifftshift = _original_ops.ifftshift
igamma = _original_ops.igamma
igammac = _original_ops.igammac
irfft2 = _original_ops.irfft2
irfftn = _original_ops.irfftn
istft = _original_ops.istft
laguerre_polynomial_l = _original_ops.laguerre_polynomial_l
legendre_polynomial_p = _original_ops.legendre_polynomial_p
lgamma = _original_ops.lgamma
logit = _original_ops.logit
logsumexp = _original_ops.logsumexp
map_coordinates = _original_ops.map_coordinates
modified_bessel_i0 = _original_ops.modified_bessel_i0
modified_bessel_i1 = _original_ops.modified_bessel_i1
modified_bessel_k0 = _original_ops.modified_bessel_k0
modified_bessel_k1 = _original_ops.modified_bessel_k1
ndtri = _original_ops.ndtri
polygamma = _original_ops.polygamma
rfft2 = _original_ops.rfft2
rfftfreq = _original_ops.rfftfreq
rfftn = _original_ops.rfftn
shifted_chebyshev_polynomial_t = _original_ops.shifted_chebyshev_polynomial_t
shifted_chebyshev_polynomial_u = _original_ops.shifted_chebyshev_polynomial_u
shifted_chebyshev_polynomial_v = _original_ops.shifted_chebyshev_polynomial_v
shifted_chebyshev_polynomial_w = _original_ops.shifted_chebyshev_polynomial_w
stft = _original_ops.stft
welch = _original_ops.welch
xlog1py = _original_ops.xlog1py
xlogy = _original_ops.xlogy
zeta = _original_ops.zeta

bessel_jn: OpDef = get_op("BesselJn")()
fft: OpDef = get_op("Fft")()
fft2: OpDef = get_op("Fft2")()
fftfreq: OpDef = get_op("Fftfreq")()
igamma_grad_a: OpDef = get_op("IgammaGradA")()
ihfft: OpDef = get_op("Ihfft")()
irfft: OpDef = get_op("Irfft")()
rfft: OpDef = get_op("Rfft")()
spence: OpDef = get_op("Spence")()

__all__ = [
    "bessel_i0e",
    "bessel_i1e",
    "bessel_jn",
    "betainc",
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "convolve",
    "convolve2d",
    "correlate",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "erfinv",
    "fft",
    "fft2",
    "fftconvolve",
    "fftfreq",
    "fftn",
    "fftshift",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "hfft",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "igamma",
    "igamma_grad_a",
    "igammac",
    "ihfft",
    "irfft",
    "irfft2",
    "irfftn",
    "istft",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
    "lgamma",
    "logit",
    "logsumexp",
    "map_coordinates",
    "modified_bessel_i0",
    "modified_bessel_i1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "ndtri",
    "polygamma",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "spence",
    "stft",
    "welch",
    "xlog1py",
    "xlogy",
    "zeta",
]
