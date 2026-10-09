"""A readable, conservative four-face scheme for tutorial-004.

Array indexing is U[j, i]: i increases to the right, j downwards.
One pixel is one finite-volume cell; no clipping is applied to intensities.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _center_gradient_reflect(S: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Centred differences with half-pixel reflected ghost cells."""
    gx = np.empty_like(S)
    gy = np.empty_like(S)
    gx[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * h)
    gx[:, 0] = (S[:, 1] - S[:, 0]) / (2.0 * h)
    gx[:, -1] = (S[:, -1] - S[:, -2]) / (2.0 * h)
    gy[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * h)
    gy[0, :] = (S[1, :] - S[0, :]) / (2.0 * h)
    gy[-1, :] = (S[-1, :] - S[-2, :]) / (2.0 * h)
    return gx, gy


def _face_coefficients(
    U: np.ndarray, *, method: str, sigma: float, kappa: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """One shared diffusivity per interior face (x, y respectively)."""
    if method == "heat":
        d = np.ones_like(U)
    elif method == "regularized":
        S = gaussian_filter(U, sigma=sigma, mode="reflect", truncate=4.0)
        gx, gy = _center_gradient_reflect(S, h)
        d = 1.0 / (1.0 + (np.hypot(gx, gy) / kappa) ** 2)
        if not np.isfinite(d).all() or np.any(d <= 0.0):
            raise FloatingPointError("diffusivity is not in (0, 1]")
    else:
        raise ValueError("method must be 'heat' or 'regularized'")
    ax = 0.5 * (d[:, :-1] + d[:, 1:])
    ay = 0.5 * (d[:-1, :] + d[1:, :])
    return ax, ay


def advance_one_step(
    U: np.ndarray,
    *,
    dt: float,
    h: float = 1.0,
    method: str = "heat",
    sigma: float = 1.5,
    kappa: float = 0.075,
) -> np.ndarray:
    """One explicit Euler step; all fluxes use the same old state U."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2 or min(U.shape) < 2 or not np.isfinite(U).all():
        raise ValueError("U must be a finite 2D image with both sizes >= 2")
    if not (np.isfinite(dt) and np.isfinite(h) and dt > 0 and h > 0):
        raise ValueError("dt and h must be positive finite numbers")
    if method not in ("heat", "regularized"):
        raise ValueError("unknown diffusion method")
    if method == "regularized" and not (
        np.isfinite(sigma) and np.isfinite(kappa) and sigma > 0 and kappa > 0
    ):
        raise ValueError("sigma and kappa must be positive finite numbers")
    r = dt / (h * h)
    if not (0.0 < r <= 0.25):
        raise ValueError("explicit 2D scheme requires 0 < dt/h^2 <= 1/4")

    ax, ay = _face_coefficients(U, method=method, sigma=sigma, kappa=kappa, h=h)
    dx = r * ax * (U[:, 1:] - U[:, :-1])
    dy = r * ay * (U[1:, :] - U[:-1, :])
    Unext = U.copy()
    Unext[:, :-1] += dx
    Unext[:, 1:] -= dx
    Unext[:-1, :] += dy
    Unext[1:, :] -= dy
    if not np.isfinite(Unext).all():
        raise FloatingPointError("nonfinite state after diffusion step")
    return Unext
