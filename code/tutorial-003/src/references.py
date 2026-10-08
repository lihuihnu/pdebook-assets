"""Independent reference solutions for tutorial-003."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import dst, idst

from heat_core import HeatGrid, analytical_eigenvalues


def smooth_initial(grid: HeatGrid) -> np.ndarray:
    return np.sin(math.pi * grid.x)


def smooth_continuous(x: np.ndarray, t: float) -> np.ndarray:
    if t < 0.0:
        raise ValueError("t must be non-negative")
    return np.exp(-(math.pi**2) * t) * np.sin(math.pi * np.asarray(x))


def box_initial(grid: HeatGrid) -> np.ndarray:
    """Frozen nodal projection of the box initial data.

    Nodes strictly inside (0.4, 0.6) receive 1, nodes outside receive 0,
    and nodes on either jump receive 1/2.
    """
    x = grid.x
    values = np.zeros_like(x)
    values[(x > 0.4) & (x < 0.6)] = 1.0
    jump = np.isclose(x, 0.4, rtol=0.0, atol=16.0 * np.finfo(float).eps) | np.isclose(
        x, 0.6, rtol=0.0, atol=16.0 * np.finfo(float).eps
    )
    values[jump] = 0.5
    return values


def semidiscrete_reference(
    grid: HeatGrid,
    initial: np.ndarray,
    t: float,
) -> np.ndarray:
    """Exact-in-time solution of U'=A_h U via orthonormal DST-I."""
    if t < 0.0:
        raise ValueError("t must be non-negative")
    state = np.asarray(initial, dtype=float)
    if state.shape != (grid.N - 1,):
        raise ValueError("initial shape does not match grid")
    coefficients = dst(state, type=1, norm="ortho")
    factors = np.exp(analytical_eigenvalues(grid) * t)
    return idst(coefficients * factors, type=1, norm="ortho")


def continuous_box_coefficients(modes: np.ndarray) -> np.ndarray:
    j = np.asarray(modes, dtype=float)
    return (2.0 / (j * math.pi)) * (
        np.cos(0.4 * j * math.pi) - np.cos(0.6 * j * math.pi)
    )


def continuous_box_reference(
    x: np.ndarray,
    t: float,
    *,
    tolerance: float = 1.0e-14,
    max_modes: int = 200_000,
) -> tuple[np.ndarray, int]:
    """Fourier-sine reference for the continuous box problem at t>0.

    The truncation is increased until a conservative single-term envelope
    4/(j*pi) * exp(-(j*pi)^2*t) falls below the requested tolerance.
    A second call with a tighter tolerance is used by later experiments when
    this reference becomes authoritative.
    """
    if t <= 0.0:
        raise ValueError("continuous box Fourier reference requires t > 0")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    j_max = 1
    while j_max < max_modes:
        envelope = (
            4.0
            / (j_max * math.pi)
            * math.exp(-((j_max * math.pi) ** 2) * t)
        )
        if envelope < tolerance and j_max >= 16:
            break
        j_max += 1
    else:
        raise RuntimeError("continuous Fourier truncation did not converge")

    modes = np.arange(1, j_max + 1, dtype=float)
    coeff = continuous_box_coefficients(modes)
    decay = np.exp(-((modes * math.pi) ** 2) * t)
    basis = np.sin(np.outer(np.asarray(x, dtype=float), modes * math.pi))
    values = basis @ (coeff * decay)
    return values, j_max
