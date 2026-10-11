"""Independent characteristic inversion and cell-average reference, smooth times only."""
import numpy as np

BREAK_TIME = 2.0 / np.pi


def u0(x):
    return 0.5 + 0.25 * np.sin(2.0 * np.pi * np.asarray(x, dtype=np.float64))


def primitive0(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x - np.cos(2.0 * np.pi * x) / (8.0 * np.pi)


def invert_smooth_characteristics(x, t):
    """Unwrapped monotone lift: xi+t*u0(xi)=x, for 0<=t<2/pi."""
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(t) and 0.0 <= t < BREAK_TIME):
        raise ValueError("smooth reference undefined at or beyond breaking")
    if t == 0.0:
        return x.copy(), 0.0
    # u0 in [1/4,3/4], hence these are guaranteed brackets.
    lo = x - 0.75 * t
    hi = x - 0.25 * t
    for _ in range(65):
        mid = 0.5 * (lo + hi)
        val = mid + t * u0(mid) - x
        lo = np.where(val < 0.0, mid, lo)
        hi = np.where(val >= 0.0, mid, hi)
    xi = 0.5 * (lo + hi)
    return xi, float(np.max(np.abs(xi + t * u0(xi) - x)))


def smooth_reference(n, t, quadrature_order=32):
    """Primitive reference with a separate physical-x Gaussian integral QA."""
    if type(n) is not int or n < 3:
        raise ValueError("n must be integer >= 3")
    faces = np.linspace(0.0, 1.0, n + 1)
    dx = 1.0 / n
    xi, residual = invert_smooth_characteristics(faces, t)
    primitive = ((primitive0(xi[1:]) - primitive0(xi[:-1]))
                 + 0.5 * t * (u0(xi[1:])**2 - u0(xi[:-1])**2)) / dx
    z, w = np.polynomial.legendre.leggauss(quadrature_order)
    sample_x = 0.5 * (faces[:-1, None] + faces[1:, None]) + 0.5 * dx * z[None, :]
    sample_xi, quad_root_residual = invert_smooth_characteristics(sample_x, t)
    quadrature = 0.5 * np.sum(w[None, :] * u0(sample_xi), axis=1)
    return {
        "exact": primitive,
        "quad": quadrature,
        "max_root_residual": max(residual, quad_root_residual),
        "max_quad_disagreement": float(np.max(np.abs(primitive - quadrature))),
        "periodic_xi_shift_error": float(abs((xi[-1] - xi[0]) - 1.0)),
    }


def exact_shock_cell_means(n, t, x_left=-1.0, x_right=1.0,
                           u_left=1.0, u_right=0.0):
    """Cell integrals of the entropy shock, never point-sampled at centers.

    Rankine--Hugoniot speed (u_left+u_right)/2. For this strict
    descending Riemann state the entropy solution is piecewise constant.
    """
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError("exact shock requires even positive cell count")
    if not (np.isfinite(t) and 0.0 <= t <= 0.5):
        raise ValueError("invalid registered Riemann time")
    if not (u_left > u_right and x_left == -1.0 and x_right == 1.0
            and u_left == 1.0 and u_right == 0.0):
        raise ValueError("ET005-02 exact reference defined only for frozen shock")
    dx = (x_right - x_left) / n
    faces = np.linspace(x_left, x_right, n + 1, dtype=np.float64)
    shock = 0.5 * (u_left + u_right) * t
    left_fraction = np.clip((shock - faces[:-1]) / dx, 0.0, 1.0)
    return u_right + (u_left - u_right) * left_fraction
