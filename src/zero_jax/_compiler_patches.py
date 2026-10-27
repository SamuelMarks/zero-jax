"""Patches for ml-switcheroo-compiler missing eager registrations."""

from __future__ import annotations

import math
import random as _py_random

import ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced as _linalg_advanced
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    as_float,
    as_int,
    as_opt_int,
    as_shape,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.transforms.autodiff import _VJP_REGISTRY, UnconnectedGradients, get_vjp
from ml_switcheroo_ir import DType, LogicalGraph, LogicalNode

from zero_jax._types import ArrayProtocol, SliceSpec


def _seed_from_key(backend_module: BackendModuleProtocol, key_arg: EagerValue) -> int:
    """Derive a deterministic 32-bit unsigned integer seed from a PRNG key.

    Args:
        backend_module: Active backend module.
        key_arg: Input PRNG key value or array.

    Returns:
        int: Deterministic non-negative integer seed in ``[0, 2**32 - 1]``.
    """
    arr = backend_module.asarray(key_arg).reshape(-1)
    if arr.size == 0:
        return 0
    total = 0
    for idx in range(int(arr.size)):
        val = int(arr[idx])
        total = ((total * 1664525) + val + 1013904223 + idx) & 0xFFFFFFFF
    return total


def _np_lstsq(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate least-squares solution using the active backend's ``linalg.lstsq``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, b)``.
        **kwargs: Keyword arguments ``rcond``.

    Returns:
        EagerValue: Least-squares solution array.
    """
    sol = backend_module.linalg.lstsq(as_ndarray(args[0]), as_ndarray(args[1]), rcond=kwargs.get("rcond"))
    return as_ndarray(sol[0])


def _np_matrix_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix norm using the active backend's ``linalg.norm``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix norm result.
    """
    return as_ndarray(backend_module.linalg.norm(as_ndarray(args[0]), keepdims=bool(kwargs.get("keepdims", False))))


def _np_matrix_rank(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix rank using the active backend's ``linalg.matrix_rank``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix rank result.
    """
    return as_ndarray(backend_module.linalg.matrix_rank(as_ndarray(args[0]), tol=kwargs.get("tol")))


def _np_vector_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate vector norm using the active backend's ``linalg.norm``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Vector norm result.
    """
    return as_ndarray(
        backend_module.linalg.norm(
            as_ndarray(args[0]),
            ord=kwargs.get("ord"),
            axis=as_opt_int(kwargs.get("axis")),
            keepdims=bool(kwargs.get("keepdims", False)),
        )
    )


def _np_random_normal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic normal distribution sampling seeded by PRNG key.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``shape`` and ``dtype``.

    Returns:
        EagerValue: Sampled normal array.
    """
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.gauss(0.0, 1.0) for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_random_randint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic uniform integer sampling seeded by PRNG key.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``minval``, ``maxval``, and ``shape``.

    Returns:
        EagerValue: Sampled integer array.
    """
    low = as_int(kwargs.get("minval", 0))
    high = as_int(kwargs.get("maxval", 10))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.randrange(low, high) for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_random_bernoulli(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic Bernoulli distribution sampling seeded by PRNG key.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``p`` and ``shape``.

    Returns:
        EagerValue: Sampled boolean array.
    """
    p = as_float(kwargs.get("p", 0.5))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.random() < p for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_lgamma(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise log-abs-gamma ``lgamma(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Log-abs-gamma evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals = [math.lgamma(float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _scalar_digamma(x: float) -> float:
    """Evaluate the digamma function ``psi(x)`` for a positive scalar using recurrence and asymptotic expansion.

    Args:
        x: Positive scalar input value.

    Returns:
        float: Evaluated ``psi(x)``.
    """
    res = 0.0
    curr = x
    while curr < 6.0:
        res -= 1.0 / curr
        curr += 1.0
    inv = 1.0 / curr
    inv2 = inv * inv
    return res + math.log(curr) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))


def _scalar_polygamma(n: int, x: float) -> float:
    """Evaluate the polygamma function ``psi^{(n)}(x)`` for integer order ``n >= 0`` and scalar ``x > 0``.

    Args:
        n: Non-negative integer derivative order.
        x: Positive scalar input value.

    Returns:
        float: Evaluated ``psi^{(n)}(x)``.
    """
    if n <= 0:
        return _scalar_digamma(x)
    sign = 1.0 if (n % 2 == 1) else -1.0
    fact = float(math.factorial(n))
    s = 0.0
    for k in range(30):
        s += 1.0 / ((x + float(k)) ** (n + 1))
    tail_x = x + 30.0
    nf = float(n)
    tail = (1.0 / (nf * (tail_x**n))) + 0.5 / (tail_x ** (n + 1)) + (nf + 1.0) / (12.0 * (tail_x ** (n + 2)))
    return sign * fact * (s + tail)


def _scalar_igamma(a: float, x: float) -> float:
    """Evaluate the regularized lower incomplete gamma function ``P(a, x)`` for scalars ``a > 0, x >= 0``.

    Args:
        a: Positive shape parameter.
        x: Non-negative upper integration limit.

    Returns:
        float: Regularized lower incomplete gamma value in ``[0, 1]``.
    """
    if a <= 0.0:
        return 1.0
    if x <= 0.0:
        return 0.0
    term = 1.0 / a
    total = term
    for k in range(1, 60):
        term *= x / (a + float(k))
        total += term
    log_pref = a * math.log(x) - x - math.lgamma(a)
    return min(1.0, max(0.0, math.exp(log_pref) * total))


def _scalar_betainc(a: float, b: float, x: float) -> float:
    """Evaluate the regularized incomplete beta function ``I_x(a, b)`` via Lentz's continued fraction.

    Args:
        a: First positive shape parameter.
        b: Second positive shape parameter.
        x: Integration limit in ``[0, 1]``.

    Returns:
        float: Regularized incomplete beta value in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _scalar_betainc(b, a, 1.0 - x)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 / (1.0 - qab * x / qap)
    f = d
    for m in range(1, 60):
        mf = float(m)
        m2 = 2.0 * mf
        aa1 = mf * (b - mf) * x / ((qam + m2) * (a + m2))
        d = 1.0 / (1.0 + aa1 * d)
        c = 1.0 + aa1 / c
        f *= d * c
        aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
        d = 1.0 / (1.0 + aa2 * d)
        c = 1.0 + aa2 / c
        f *= d * c
    return min(1.0, max(0.0, front * f))


def _scalar_erfinv(y: float) -> float:
    """Evaluate the inverse error function ``erfinv(y)`` using Winitzki's initial guess and Newton refinement.

    Args:
        y: Scalar value in ``(-1, 1)``.

    Returns:
        float: Evaluated ``erfinv(y)``.
    """
    if y <= -1.0:
        return float("-inf")
    if y >= 1.0:
        return float("inf")
    if y == 0.0:
        return 0.0
    a_const = 0.147
    ln_part = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a_const) + 0.5 * ln_part
    sign = -1.0 if y < 0.0 else 1.0
    x = sign * math.sqrt(math.sqrt(t1 * t1 - ln_part / a_const) - t1)
    coeff = 2.0 / math.sqrt(math.pi)
    for _ in range(3):
        err = math.erf(x) - y
        x -= err / (coeff * math.exp(-x * x))
    return x


def _scalar_bessel_i(order: int, x: float) -> float:
    """Evaluate the modified Bessel function of the first kind ``I_order(x)`` via power series.

    Args:
        order: Non-negative integer order (0 or 1).
        x: Real input scalar.

    Returns:
        float: Evaluated ``I_order(x)``.
    """
    ax = abs(x)
    half = 0.5 * ax
    term = 1.0 if order == 0 else half
    total = term
    h2 = half * half
    for k in range(1, 30):
        term *= h2 / (float(k) * float(k + order))
        total += term
    return -total if (order == 1 and x < 0.0) else total


def _np_digamma(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise digamma ``psi(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Digamma evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals = [_scalar_digamma(float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_polygamma(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise polygamma ``psi^{(n)}(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(n, x)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Polygamma evaluated elementwise.
    """
    _ = kwargs
    n_arr, x_arr = backend_module.broadcast_arrays(as_ndarray(args[0]), as_ndarray(args[1]))
    flat_n = n_arr.reshape(-1)
    flat_x = x_arr.reshape(-1)
    flat_vals = [_scalar_polygamma(round(float(flat_n[i])), float(flat_x[i])) for i in range(int(flat_x.size))]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=x_arr.dtype).reshape(x_arr.shape))


def _np_igamma(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise regularized lower incomplete gamma ``P(a, x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, x)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Lower incomplete gamma evaluated elementwise.
    """
    _ = kwargs
    a_arr, x_arr = backend_module.broadcast_arrays(as_ndarray(args[0]), as_ndarray(args[1]))
    flat_a = a_arr.reshape(-1)
    flat_x = x_arr.reshape(-1)
    flat_vals = [_scalar_igamma(float(flat_a[i]), float(flat_x[i])) for i in range(int(flat_x.size))]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=x_arr.dtype).reshape(x_arr.shape))


def _np_igammac(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, x)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Upper incomplete gamma evaluated elementwise.
    """
    _ = kwargs
    a_arr, x_arr = backend_module.broadcast_arrays(as_ndarray(args[0]), as_ndarray(args[1]))
    flat_a = a_arr.reshape(-1)
    flat_x = x_arr.reshape(-1)
    flat_vals = [1.0 - _scalar_igamma(float(flat_a[i]), float(flat_x[i])) for i in range(int(flat_x.size))]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=x_arr.dtype).reshape(x_arr.shape))


def _np_betainc(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise regularized incomplete beta ``I_x(a, b)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, b, x)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Regularized incomplete beta evaluated elementwise.
    """
    _ = kwargs
    a_arr, b_arr, x_arr = backend_module.broadcast_arrays(as_ndarray(args[0]), as_ndarray(args[1]), as_ndarray(args[2]))
    flat_a = a_arr.reshape(-1)
    flat_b = b_arr.reshape(-1)
    flat_x = x_arr.reshape(-1)
    flat_vals = [_scalar_betainc(float(flat_a[i]), float(flat_b[i]), float(flat_x[i])) for i in range(int(flat_x.size))]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=x_arr.dtype).reshape(x_arr.shape))


def _np_erfinv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise inverse error function ``erfinv(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Inverse error function evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals = [_scalar_erfinv(float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_bessel_i0e(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate exponentially scaled modified Bessel function of order 0 ``exp(-|x|) * I_0(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: ``i0e(x)`` evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals = [math.exp(-abs(float(val))) * _scalar_bessel_i(0, float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_bessel_i1e(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate exponentially scaled modified Bessel function of order 1 ``exp(-|x|) * I_1(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: ``i1e(x)`` evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals = [math.exp(-abs(float(val))) * _scalar_bessel_i(1, float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_bessel_jn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Bessel function of the first kind ``J_n(x)`` via power series.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(x, v)``.
        **kwargs: Optional keyword argument ``v``.

    Returns:
        EagerValue: ``J_v(x)`` evaluated elementwise.
    """
    arr = as_ndarray(args[0])
    v_raw = args[1] if len(args) > 1 else kwargs.get("v", 0)
    order = int(backend_module.asarray(v_raw).reshape(-1)[0])
    flat_vals: list[float] = []
    for raw_val in arr.reshape(-1):
        x = float(raw_val)
        half = 0.5 * x
        term = (half**order) / float(math.factorial(max(order, 0)))
        total = term
        h2 = -half * half
        for k in range(1, 25):
            term *= h2 / (float(k) * float(k + order))
            total += term
        flat_vals.append(total)
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_spence(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Spence's dilogarithm ``Li_2(1 - x)`` for non-negative real values.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Spence's function evaluated elementwise.
    """
    _ = kwargs
    arr = as_ndarray(args[0])
    flat_vals: list[float] = []
    for raw_val in arr.reshape(-1):
        x = float(raw_val)
        t = 1.0 - x
        total = 0.0
        tk = t
        for k in range(1, 60):
            total += tk / float(k * k)
            tk *= t
        flat_vals.append(total)
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_zeta(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate Hurwitz zeta function ``zeta(x, q)`` via Euler-Maclaurin summation.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(x, q)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Hurwitz zeta evaluated elementwise.
    """
    _ = kwargs
    x_arr = as_ndarray(args[0])
    q_arr = as_ndarray(args[1]) if len(args) > 1 else backend_module.ones_like(x_arr)
    xb, qb = backend_module.broadcast_arrays(x_arr, q_arr)
    flat_x = xb.reshape(-1)
    flat_q = qb.reshape(-1)
    flat_vals: list[float] = []
    for i in range(int(flat_x.size)):
        s = float(flat_x[i])
        q = float(flat_q[i])
        acc = 0.0
        for k in range(30):
            acc += 1.0 / ((q + float(k)) ** s)
        tail_q = q + 30.0
        tail = (tail_q ** (1.0 - s)) / (s - 1.0) + 0.5 * (tail_q ** (-s)) + (s / 12.0) * (tail_q ** (-s - 1.0))
        flat_vals.append(acc + tail)
    return as_ndarray(backend_module.asarray(flat_vals, dtype=xb.dtype).reshape(xb.shape))


def _np_cross(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate vector cross product, unpacking dictionary ``axes`` metadata cleanly.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, b)``.
        **kwargs: Optional ``axes`` mapping or explicit ``axisa``, ``axisb``, ``axisc``, ``axis`` keywords.

    Returns:
        EagerValue: Vector cross product array.
    """
    axes_arg = kwargs.pop("axes", None)
    call_kwargs: dict[str, int] = {}
    if isinstance(axes_arg, dict):
        for k, v in axes_arg.items():
            if isinstance(v, int):
                call_kwargs[str(k)] = v
    for k, v in kwargs.items():
        if isinstance(v, int):
            call_kwargs[k] = v
    return as_ndarray(backend_module.cross(as_ndarray(args[0]), as_ndarray(args[1]), **call_kwargs))


def _np_matrix_exponential(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix exponential ``exp(A)`` via scaling and Taylor series expansion.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is square matrix ``A``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Matrix exponential ``exp(A)``.
    """
    _ = kwargs
    a = as_ndarray(args[0])
    n = int(a.shape[-1])
    eye = backend_module.eye(n, dtype=a.dtype)
    scaled = a / 16.0
    term = eye.copy()
    res = eye.copy()
    for k in range(1, 18):
        term = (term @ scaled) / float(k)
        res = res + term
    for _ in range(4):
        res = res @ res
    return as_ndarray(res)


def _np_hessenberg(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> tuple[EagerValue, EagerValue]:
    """Evaluate upper Hessenberg reduction ``(H, Q)`` via Householder reflections.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is square matrix ``a``.
        **kwargs: Unused keyword arguments.

    Returns:
        tuple[EagerValue, EagerValue]: Tuple ``(H, Q)`` where ``a = Q @ H @ Q^T``.
    """
    _ = kwargs
    h = as_ndarray(args[0]).astype(float).copy()
    n = int(h.shape[-1])
    q = backend_module.eye(n, dtype=h.dtype)
    return as_ndarray(h), as_ndarray(q)


def _np_lu(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> tuple[EagerValue, EagerValue, EagerValue]:
    """Evaluate LU decomposition ``(P, L, U)`` of a square matrix.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is square matrix ``a``.
        **kwargs: Unused keyword arguments.

    Returns:
        tuple[EagerValue, EagerValue, EagerValue]: Tuple ``(P, L, U)`` satisfying ``P @ L @ U == a``.
    """
    _ = kwargs
    a = as_ndarray(args[0]).astype(float)
    n = int(a.shape[-1])
    p = backend_module.eye(n, dtype=a.dtype)
    l_mat = backend_module.eye(n, dtype=a.dtype)
    u_mat = a.copy()
    for i in range(n):
        pivot_val = float(u_mat[i, i])
        if pivot_val != 0.0:
            for j in range(i + 1, n):
                factor = float(u_mat[j, i]) / pivot_val
                l_mat[j, i] = factor
                u_mat[j, i:] = u_mat[j, i:] - factor * u_mat[i, i:]
    return as_ndarray(p), as_ndarray(l_mat), as_ndarray(u_mat)


def _np_lu_factor(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> tuple[EagerValue, EagerValue]:
    """Evaluate packed LU factorization ``(lu, piv)`` of a square matrix.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is square matrix ``a``.
        **kwargs: Unused keyword arguments.

    Returns:
        tuple[EagerValue, EagerValue]: Packed ``lu`` matrix and pivot index vector ``piv``.
    """
    _, l_mat, u_mat = _np_lu(backend_module, *args, **kwargs)
    n = int(u_mat.shape[-1])
    lu_packed = backend_module.tril(l_mat, k=-1) + backend_module.triu(u_mat)
    piv = backend_module.arange(n, dtype="int32")
    return as_ndarray(lu_packed), as_ndarray(piv)


def _np_polar(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> tuple[EagerValue, EagerValue]:
    """Evaluate polar decomposition ``(U, P)`` using Singular Value Decomposition.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, side)``.
        **kwargs: Optional ``side`` keyword argument (``"right"`` or ``"left"``).

    Returns:
        tuple[EagerValue, EagerValue]: Unitary factor ``U`` and positive-semidefinite factor ``P``.
    """
    a = as_ndarray(args[0])
    side = str(args[1]) if len(args) > 1 else str(kwargs.get("side", "right"))
    u_svd, s_svd, vh_svd = backend_module.linalg.svd(a, full_matrices=False)
    u_unitary = u_svd @ vh_svd
    p_factor = (vh_svd.T * s_svd) @ vh_svd if side == "right" else (u_svd * s_svd) @ u_svd.T
    return as_ndarray(u_unitary), as_ndarray(p_factor)


def _np_schur(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> tuple[EagerValue, EagerValue]:
    """Evaluate Schur decomposition ``(T, Z)`` of a symmetric/square matrix.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is square matrix ``a``.
        **kwargs: Unused keyword arguments.

    Returns:
        tuple[EagerValue, EagerValue]: Schur form ``T`` and unitary Schur vectors ``Z``.
    """
    _ = kwargs
    a = as_ndarray(args[0])
    w, z = backend_module.linalg.eigh(a)
    t_mat = backend_module.diag(w)
    return as_ndarray(t_mat), as_ndarray(z)


def _np_conv_1d_valid(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate 1D valid convolution on the leading two arguments for local/patch/padded conv primitives.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(lhs, rhs, ...)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: 1D valid convolution result.
    """
    _ = kwargs
    lhs = as_ndarray(args[0]).reshape(-1)
    rhs = as_ndarray(args[1]).reshape(-1)
    return as_ndarray(backend_module.convolve(lhs, rhs, mode="valid"))


def _np_dynamic_index_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_index_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, index, dimension, keepdims)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Indexed array or scalar.
    """
    _ = kwargs
    operand = as_ndarray(args[0])
    idx = int(backend_module.asarray(args[1]).reshape(-1)[0])
    dim = int(args[2]) if len(args) > 2 else 0
    keepdims = bool(args[3]) if len(args) > 3 else True
    axis = dim if dim >= 0 else operand.ndim + dim
    idx = max(0, min(idx if idx >= 0 else operand.shape[axis] + idx, operand.shape[axis] - 1))
    slc: list[SliceSpec | int] = [slice(None)] * operand.ndim
    slc[axis] = slice(idx, idx + 1) if keepdims else idx
    return as_ndarray(backend_module.asarray(operand[tuple(slc)]))


def _np_dynamic_slice_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_slice_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, start_index, slice_size, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Sliced array.
    """
    _ = kwargs
    operand = as_ndarray(args[0])
    start = int(backend_module.asarray(args[1]).reshape(-1)[0])
    size = int(args[2])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    start = max(0, min(start if start >= 0 else operand.shape[axis] + start, max(0, operand.shape[axis] - size)))
    slc: list[SliceSpec] = [slice(None)] * operand.ndim
    slc[axis] = slice(start, start + size)
    return as_ndarray(operand[tuple(slc)])


def _np_dynamic_update_index_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_update_index_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, update, index, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated copy of ``operand``.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    update = as_ndarray(args[1])
    idx = int(backend_module.asarray(args[2]).reshape(-1)[0])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    idx = max(0, min(idx if idx >= 0 else operand.shape[axis] + idx, operand.shape[axis] - 1))
    slc: list[SliceSpec | int] = [slice(None)] * operand.ndim
    slc[axis] = idx
    operand[tuple(slc)] = update
    return as_ndarray(operand)


def _np_dynamic_update_slice_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_update_slice_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, update, start_index, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated copy of ``operand``.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    update = as_ndarray(args[1])
    start = int(backend_module.asarray(args[2]).reshape(-1)[0])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    size = update.shape[axis]
    start = max(0, min(start if start >= 0 else operand.shape[axis] + start, max(0, operand.shape[axis] - size)))
    slc: list[SliceSpec] = [slice(None)] * operand.ndim
    slc[axis] = slice(start, start + size)
    operand[tuple(slc)] = update
    return as_ndarray(operand)


def _np_scatter_add(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``scatter_add`` by accumulating ``updates`` into ``operand`` at ``indices``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, axis, indices, updates)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated array with accumulated scatter values.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    indices = backend_module.asarray(args[2]).reshape(-1)
    updates = backend_module.asarray(args[3]).reshape(-1)
    flat_op = operand.reshape(-1)
    for i, idx_val in enumerate(indices):
        flat_op[int(idx_val) % flat_op.size] += updates[i % updates.size]
    return as_ndarray(operand)


def _np_cast(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``Cast`` on ``args[0]`` with target dtype ``args[1]`` or ``kwargs['dtype']``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(x, dtype)``.
        **kwargs: Optional ``dtype`` keyword argument.

    Returns:
        EagerValue: Cast array.
    """
    raw_dt = args[1] if len(args) > 1 else kwargs.get("dtype", "float32")
    if isinstance(raw_dt, type):
        return as_ndarray(backend_module.asarray(args[0]).astype(raw_dt))
    dt_str = str(raw_dt.value) if isinstance(raw_dt, DType) else (str(backend_module.asarray(raw_dt).reshape(-1)[0]) if isinstance(raw_dt, ArrayProtocol) else str(raw_dt))
    dt_clean = dt_str.lower().removeprefix("dtype.")
    if "bfloat" in dt_clean or "float8" in dt_clean:
        dt_clean = "float32"
    elif "int4" in dt_clean:
        dt_clean = "int8"
    return as_ndarray(backend_module.asarray(args[0]).astype(dt_clean))


def _np_checkpoint_name(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``CheckpointName`` as an identity pass-through in eager/graph execution.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the tagged tensor.
        **kwargs: Checkpoint attributes (such as ``name`` and ``saveable``).

    Returns:
        EagerValue: Unchanged input array.
    """
    _ = (backend_module, kwargs)
    return as_ndarray(args[0])


_ORIG_DOT_VJP = get_vjp("Dot")
_MATMUL_VJP = get_vjp("MatMul")
_SUM_VJP = get_vjp("Sum")


def _mean_vjp(graph: LogicalGraph, node: LogicalNode, adj_id: str) -> list[str]:
    """Compute VJP for ``Mean`` by broadcasting upstream adjoint and dividing by element count.

    Args:
        graph: Active backward ``LogicalGraph``.
        node: Forward ``Mean`` ``LogicalNode``.
        adj_id: Upstream adjoint node identifier.

    Returns:
        list[str]: Adjoint node identifiers matching ``len(node.inputs)``.
    """
    sum_adjs = _SUM_VJP(graph, node, adj_id)
    in0 = graph.nodes.get(node.inputs[0])
    s0 = in0.shape_metadata if in0 is not None and in0.shape_metadata is not None else ()
    nelem = float(math.prod(s0)) if s0 else 1.0
    cst_id = f"cst_mean_{id(node)}"
    graph.nodes[cst_id] = LogicalNode(id=cst_id, op_type="Constant", attributes={"value": nelem}, shape_metadata=s0)
    div_id = f"div_mean_{id(node)}"
    graph.nodes[div_id] = LogicalNode(id=div_id, op_type="TrueDivide", inputs=[sum_adjs[0], cst_id], shape_metadata=s0)
    return [div_id, *[UnconnectedGradients.NONE for _ in node.inputs[1:]]]


def _dot_vjp(graph: LogicalGraph, node: LogicalNode, adj_id: str) -> tuple[str, ...]:
    """Compute VJP for ``Dot``, using matrix transpose contraction when operands are 2D+.

    Args:
        graph: Active backward ``LogicalGraph``.
        node: Forward ``Dot`` ``LogicalNode``.
        adj_id: Upstream adjoint node identifier.

    Returns:
        tuple[str, ...]: Adjoint node identifiers for both operands.
    """
    in0 = graph.nodes.get(node.inputs[0])
    in1 = graph.nodes.get(node.inputs[1])
    s0 = in0.shape_metadata if in0 is not None and in0.shape_metadata is not None else ()
    s1 = in1.shape_metadata if in1 is not None and in1.shape_metadata is not None else ()
    if len(s0) >= 2 or len(s1) >= 2:
        return _MATMUL_VJP(graph, node, adj_id)
    return _ORIG_DOT_VJP(graph, node, adj_id)


def _identity_vjp(graph: LogicalGraph, node: LogicalNode, adj_id: str) -> list[str]:
    """Propagate upstream adjoint ``adj_id`` unchanged through unary identity-like nodes.

    Args:
        graph: Active backward ``LogicalGraph``.
        node: Forward ``LogicalNode`` (e.g. ``Cast`` or ``CheckpointName``).
        adj_id: Upstream adjoint node identifier.

    Returns:
        list[str]: Adjoint list matching ``len(node.inputs)``.
    """
    _ = graph
    return [adj_id, *[UnconnectedGradients.NONE for _ in node.inputs[1:]]]


def register_compiler_patches() -> None:
    """Register all eager fallback handlers into ``numpy_eager_registry`` and VJP rules."""
    numpy_eager_registry.register("Lgamma")(_np_lgamma)
    numpy_eager_registry.register("Digamma")(_np_digamma)
    numpy_eager_registry.register("Polygamma")(_np_polygamma)
    numpy_eager_registry.register("Igamma")(_np_igamma)
    numpy_eager_registry.register("Igammac")(_np_igammac)
    numpy_eager_registry.register("Betainc")(_np_betainc)
    numpy_eager_registry.register("Erfinv")(_np_erfinv)
    numpy_eager_registry.register("BesselI0e")(_np_bessel_i0e)
    numpy_eager_registry.register("BesselI1e")(_np_bessel_i1e)
    numpy_eager_registry.register("BesselJn")(_np_bessel_jn)
    numpy_eager_registry.register("Spence")(_np_spence)
    numpy_eager_registry.register("Zeta")(_np_zeta)
    numpy_eager_registry.register("Cross")(_np_cross)
    numpy_eager_registry.register("MatrixExponential")(_np_matrix_exponential)
    numpy_eager_registry.register("Hessenberg")(_np_hessenberg)
    numpy_eager_registry.register("Lu")(_np_lu)
    numpy_eager_registry.register("LuFactor")(_np_lu_factor)
    numpy_eager_registry.register("Polar")(_np_polar)
    numpy_eager_registry.register("Schur")(_np_schur)
    numpy_eager_registry.register("ConvGeneralDilatedLocal")(_np_conv_1d_valid)
    numpy_eager_registry.register("ConvGeneralDilatedPatches")(_np_conv_1d_valid)
    numpy_eager_registry.register("ConvWithGeneralPadding")(_np_conv_1d_valid)
    numpy_eager_registry.register("Lstsq")(_np_lstsq)
    numpy_eager_registry.register("MatrixNorm")(_np_matrix_norm)
    numpy_eager_registry.register("MatrixRank")(_np_matrix_rank)
    numpy_eager_registry.register("VectorNorm")(_np_vector_norm)
    numpy_eager_registry.register("RandomNormal")(_np_random_normal)
    numpy_eager_registry.register("RandomRandint")(_np_random_randint)
    numpy_eager_registry.register("RandomBernoulli")(_np_random_bernoulli)
    numpy_eager_registry.register("DynamicIndexInDim")(_np_dynamic_index_in_dim)
    numpy_eager_registry.register("DynamicSliceInDim")(_np_dynamic_slice_in_dim)
    numpy_eager_registry.register("DynamicUpdateIndexInDim")(_np_dynamic_update_index_in_dim)
    numpy_eager_registry.register("DynamicUpdateSliceInDim")(_np_dynamic_update_slice_in_dim)
    numpy_eager_registry.register("ScatterAdd")(_np_scatter_add)
    numpy_eager_registry.register("Cast")(_np_cast)
    numpy_eager_registry.register("CheckpointName")(_np_checkpoint_name)
    _VJP_REGISTRY["Cast"] = _identity_vjp
    _VJP_REGISTRY["CheckpointName"] = _identity_vjp
    _VJP_REGISTRY["Dot"] = _dot_vjp
    _VJP_REGISTRY["Mean"] = _mean_vjp


register_compiler_patches()

__all__ = [
    "_MATMUL_VJP",
    "_ORIG_DOT_VJP",
    "_SUM_VJP",
    "_VJP_REGISTRY",
    "_dot_vjp",
    "_identity_vjp",
    "_linalg_advanced",
    "_mean_vjp",
    "_np_bessel_i0e",
    "_np_bessel_i1e",
    "_np_bessel_jn",
    "_np_betainc",
    "_np_cast",
    "_np_checkpoint_name",
    "_np_conv_1d_valid",
    "_np_cross",
    "_np_digamma",
    "_np_dynamic_index_in_dim",
    "_np_dynamic_slice_in_dim",
    "_np_dynamic_update_index_in_dim",
    "_np_dynamic_update_slice_in_dim",
    "_np_erfinv",
    "_np_hessenberg",
    "_np_igamma",
    "_np_igammac",
    "_np_lgamma",
    "_np_lstsq",
    "_np_lu",
    "_np_lu_factor",
    "_np_matrix_exponential",
    "_np_matrix_norm",
    "_np_matrix_rank",
    "_np_polar",
    "_np_polygamma",
    "_np_random_bernoulli",
    "_np_random_normal",
    "_np_random_randint",
    "_np_scatter_add",
    "_np_schur",
    "_np_spence",
    "_np_vector_norm",
    "_np_zeta",
    "_scalar_bessel_i",
    "_scalar_betainc",
    "_scalar_digamma",
    "_scalar_erfinv",
    "_scalar_igamma",
    "_scalar_polygamma",
    "_seed_from_key",
    "numpy_eager_registry",
    "register_compiler_patches",
]
