"""Transparent second-order wave solver for tutorial-006.

Only the interior three-level stencil and the two endpoint updates advance the
solution.  L2 displacement diagnostics are not a physical or discrete energy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def pulse(x: np.ndarray, center: float = 0.35,
          half_width: float = 0.1, power: int = 6) -> np.ndarray:
    """Compact polynomial pulse; zero outside its open support."""
    x = np.asarray(x, dtype=float)
    if half_width <= 0 or power < 2:
        raise ValueError("half_width must be positive and power at least 2")
    z = (x - center) / half_width
    inside = np.abs(z) < 1.0
    result = np.zeros_like(x)
    result[inside] = (1.0 - z[inside] ** 2) ** power
    return result


def pulse_derivative(x: np.ndarray, center: float = 0.35,
                     half_width: float = 0.1, power: int = 6) -> np.ndarray:
    """Analytic first derivative, including the zero extension."""
    x = np.asarray(x, dtype=float)
    if half_width <= 0 or power < 2:
        raise ValueError("half_width must be positive and power at least 2")
    z = (x - center) / half_width
    inside = np.abs(z) < 1.0
    result = np.zeros_like(x)
    result[inside] = (-2.0 * power / half_width * z[inside]
                      * (1.0 - z[inside] ** 2) ** (power - 1))
    return result


def discrete_inner(a: np.ndarray, b: np.ndarray, h: float) -> float:
    """Composite-trapezoid inner product on a nodal grid."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or h <= 0:
        raise ValueError("equal one-dimensional nodal arrays and h > 0 required")
    product = a * b
    return float(h * (0.5 * product[0] + np.sum(product[1:-1])
                      + 0.5 * product[-1]))


def displacement_l2(values: np.ndarray, h: float) -> float:
    return math.sqrt(max(0.0, discrete_inner(values, values, h)))


def _check_courant(courant: float) -> None:
    if not math.isfinite(courant) or not 0.0 < courant <= 1.0:
        raise ValueError("the interior Courant number must satisfy 0 < C <= 1")


def _check_states(first: np.ndarray, second: np.ndarray) -> None:
    if (first.ndim != 1 or first.shape != second.shape or first.size < 3
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))):
        raise ValueError("finite one-dimensional states of equal size >= 3 required")


def _set_endpoints(new: np.ndarray, current: np.ndarray,
                   boundary: str, boundary_courant: float | None) -> None:
    """Use the already-computed new interior; never an in-place time stencil."""
    if boundary == "dirichlet":
        new[0] = 0.0
        new[-1] = 0.0
    elif boundary == "mur":
        if (boundary_courant is None
                or not math.isfinite(boundary_courant)
                or boundary_courant <= 0.0):
            raise ValueError("Mur requires a finite positive boundary Courant number")
        q = (boundary_courant - 1.0) / (boundary_courant + 1.0)
        new[0] = current[1] + q * (new[1] - current[0])
        new[-1] = current[-2] + q * (new[-2] - current[-1])
    else:
        raise ValueError("boundary must be 'dirichlet' or 'mur'")


def initial_step(displacement: np.ndarray, velocity: np.ndarray,
                 courant: float, dt: float, boundary: str = "dirichlet",
                 boundary_courant: float | None = None) -> np.ndarray:
    """Taylor start with the centered discrete Laplacian, then endpoints."""
    displacement = np.asarray(displacement, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    _check_states(displacement, velocity)
    _check_courant(courant)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    new = np.empty_like(displacement)
    new[1:-1] = (displacement[1:-1] + dt * velocity[1:-1]
                 + 0.5 * courant ** 2
                 * (displacement[2:] - 2.0 * displacement[1:-1]
                    + displacement[:-2]))
    _set_endpoints(new, displacement, boundary, boundary_courant)
    return new


def leapfrog_step(previous: np.ndarray, current: np.ndarray,
                  courant: float, boundary: str = "dirichlet",
                  boundary_courant: float | None = None) -> np.ndarray:
    """Advance n-1,n -> n+1; endpoint formulas use new interior nodes."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    _check_states(previous, current)
    _check_courant(courant)
    new = np.empty_like(current)
    new[1:-1] = (2.0 * current[1:-1] - previous[1:-1]
                 + courant ** 2
                 * (current[2:] - 2.0 * current[1:-1] + current[:-2]))
    _set_endpoints(new, current, boundary, boundary_courant)
    return new


def integer_step(time: float, dt: float) -> int:
    """Reject incompatible times instead of shortening a three-level step."""
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and nonnegative")
    step = int(round(time / dt))
    if not math.isclose(time, step * dt, rel_tol=0.0,
                        abs_tol=64.0 * np.finfo(float).eps * max(1.0, time)):
        raise ValueError(f"time {time} is not an integer multiple of dt={dt}")
    return step


@dataclass
class WaveResult:
    x: np.ndarray
    snapshots: dict[int, np.ndarray]
    diagnostics: list[tuple[int, float, float, float, float]]
    h: float
    dt: float
    final_step: int
    crop_start: int
    full_nodes: int


def solve_wave(n: int, *, domain: Sequence[int] = (0, 1),
               c: float = 1.0, courant: float = 0.8,
               final_time: float = 1.2,
               snapshot_times: Sequence[float] = (0.0, 0.4, 0.6, 0.9, 1.2),
               boundary: str = "dirichlet",
               boundary_speed_ratio: float | None = None,
               center: float = 0.35, half_width: float = 0.1,
               power: int = 6, crop: Sequence[int] = (0, 1)) -> WaveResult:
    """Solve on an integer-endpoint domain with h=1/n.

    The large-domain run uses the same integer global grid indices as the short
    run.  Its crop is therefore both geometrically exact and bitwise identical
    in coordinates, with offset 2*n for domain=(-2,3), crop=(0,1).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    _check_courant(courant)
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("c must be finite and positive")
    if len(domain) != 2 or len(crop) != 2:
        raise ValueError("domain and crop require two endpoints")
    if any(isinstance(v, bool) or int(v) != v for v in (*domain, *crop)):
        raise ValueError("this solver requires integer domain and crop endpoints")
    left, right = map(int, domain)
    crop_left, crop_right = map(int, crop)
    if not left <= crop_left < crop_right <= right:
        raise ValueError("crop must lie inside the increasing domain")
    if boundary not in ("dirichlet", "mur"):
        raise ValueError("unknown boundary")
    if boundary == "mur" and (boundary_speed_ratio is None
            or not math.isfinite(boundary_speed_ratio)
            or boundary_speed_ratio <= 0.0):
        raise ValueError("Mur boundary_speed_ratio must be finite and positive")
    h = 1.0 / n
    dt = courant * h / c
    final_step = integer_step(final_time, dt)
    requested_steps = [integer_step(float(t), dt) for t in snapshot_times]
    if any(k > final_step for k in requested_steps):
        raise ValueError("snapshot exceeds final_time")
    if len(set(requested_steps)) != len(requested_steps):
        raise ValueError("snapshot times must give distinct steps")
    wanted = set(requested_steps)
    x_full = np.arange(left * n, right * n + 1, dtype=float) / n
    crop_start = (crop_left - left) * n
    crop_stop = (crop_right - left) * n + 1
    selection = slice(crop_start, crop_stop)
    x = x_full[selection].copy()
    previous = pulse(x_full, center, half_width, power)
    velocity = -c * pulse_derivative(x_full, center, half_width, power)
    if previous[0] != 0.0 or previous[-1] != 0.0:
        raise ValueError("the selected pulse must vanish at both domain endpoints")
    boundary_courant = (None if boundary == "dirichlet"
                        else courant * boundary_speed_ratio)
    snapshots: dict[int, np.ndarray] = {}
    diagnostics: list[tuple[int, float, float, float, float]] = []

    def record(step: int, state: np.ndarray) -> None:
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"non-finite state at step {step}")
        diagnostics.append((step, step * dt, float(np.min(state)),
                            float(np.max(state)), displacement_l2(state, h)))
        if step in wanted:
            snapshots[step] = state[selection].copy()

    record(0, previous)
    if final_step == 0:
        return WaveResult(x, snapshots, diagnostics, h, dt, final_step,
                          crop_start, x_full.size)
    current = initial_step(previous, velocity, courant, dt,
                           boundary, boundary_courant)
    record(1, current)
    for step in range(1, final_step):
        new = leapfrog_step(previous, current, courant,
                            boundary, boundary_courant)
        previous, current = current, new
        record(step + 1, current)
    return WaveResult(x, snapshots, diagnostics, h, dt, final_step,
                      crop_start, x_full.size)
