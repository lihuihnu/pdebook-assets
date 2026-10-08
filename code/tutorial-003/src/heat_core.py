"""Shared finite-difference/time-stepping core for tutorial-003.

The module intentionally exposes the semidiscrete heat operator and all three
time integrators used by the article contract.  SciPy is used only for sparse
linear algebra; no PDE package is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class HeatGrid:
    N: int
    h: float
    x: np.ndarray


@dataclass(frozen=True)
class AdvanceResult:
    values: np.ndarray
    linear_solves: int


def make_grid(N: int) -> HeatGrid:
    """Return the frozen uniform grid with N intervals and N-1 unknowns."""
    if N < 2:
        raise ValueError("N must be >= 2")
    h = 1.0 / float(N)
    x = np.arange(1, N, dtype=float) * h
    return HeatGrid(N=N, h=h, x=x)


def discrete_laplacian(grid: HeatGrid) -> csc_matrix:
    """Return the Dirichlet second-difference matrix A_h."""
    n = grid.N - 1
    inv_h2 = 1.0 / (grid.h * grid.h)
    main = np.full(n, -2.0 * inv_h2, dtype=float)
    off = np.full(max(n - 1, 0), inv_h2, dtype=float)
    return diags((off, main, off), offsets=(-1, 0, 1), format="csc")


def analytical_eigenvalue(grid: HeatGrid, mode: int) -> float:
    """Analytical eigenvalue for discrete sine mode j."""
    if not 1 <= mode <= grid.N - 1:
        raise ValueError("mode must be in [1, N-1]")
    angle = mode * math.pi / (2.0 * grid.N)
    return -4.0 / (grid.h * grid.h) * math.sin(angle) ** 2


def analytical_eigenvalues(grid: HeatGrid) -> np.ndarray:
    return np.asarray(
        [analytical_eigenvalue(grid, j) for j in range(1, grid.N)],
        dtype=float,
    )


def sine_mode(grid: HeatGrid, mode: int, *, normalize: bool = True) -> np.ndarray:
    """Return v_j(i)=sin(i*j*pi/N) on interior nodes."""
    if not 1 <= mode <= grid.N - 1:
        raise ValueError("mode must be in [1, N-1]")
    indices = np.arange(1, grid.N, dtype=float)
    values = np.sin(indices * mode * math.pi / grid.N)
    if normalize:
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            raise RuntimeError("zero sine-mode norm")
        values = values / norm
    return values


def amplification_backward_euler(xi: float | np.ndarray):
    return 1.0 / (1.0 + np.asarray(xi))


def amplification_crank_nicolson(xi: float | np.ndarray):
    xi_array = np.asarray(xi)
    return (1.0 - 0.5 * xi_array) / (1.0 + 0.5 * xi_array)


def amplification_rannacher_startup(xi: float | np.ndarray):
    """Two backward-Euler half steps spanning one full interval."""
    return 1.0 / (1.0 + 0.5 * np.asarray(xi)) ** 2


def _validate_state(matrix: csc_matrix, values: np.ndarray) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.ndim != 1 or state.shape[0] != matrix.shape[0]:
        raise ValueError("state shape does not match heat operator")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains non-finite values")
    return state.copy()


def advance_backward_euler(
    matrix: csc_matrix,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> AdvanceResult:
    if dt <= 0.0 or steps < 0:
        raise ValueError("dt must be positive and steps non-negative")
    state = _validate_state(matrix, initial)
    if steps == 0:
        return AdvanceResult(values=state, linear_solves=0)

    ident = eye(matrix.shape[0], format="csc")
    solver = splu((ident - dt * matrix).tocsc())
    for _ in range(steps):
        state = solver.solve(state)
    return AdvanceResult(values=state, linear_solves=steps)


def advance_crank_nicolson(
    matrix: csc_matrix,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> AdvanceResult:
    if dt <= 0.0 or steps < 0:
        raise ValueError("dt must be positive and steps non-negative")
    state = _validate_state(matrix, initial)
    if steps == 0:
        return AdvanceResult(values=state, linear_solves=0)

    ident = eye(matrix.shape[0], format="csc")
    left = (ident - 0.5 * dt * matrix).tocsc()
    right = (ident + 0.5 * dt * matrix).tocsc()
    solver = splu(left)
    for _ in range(steps):
        state = solver.solve(right @ state)
    return AdvanceResult(values=state, linear_solves=steps)


def advance_rannacher_r1(
    matrix: csc_matrix,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> AdvanceResult:
    """R1: two BE half steps over [0, dt], then CN full steps."""
    if dt <= 0.0 or steps < 1:
        raise ValueError("R1 requires dt > 0 and at least one full interval")
    state = _validate_state(matrix, initial)

    ident = eye(matrix.shape[0], format="csc")
    half_solver = splu((ident - 0.5 * dt * matrix).tocsc())
    state = half_solver.solve(state)
    state = half_solver.solve(state)
    solves = 2

    if steps > 1:
        left = (ident - 0.5 * dt * matrix).tocsc()
        right = (ident + 0.5 * dt * matrix).tocsc()
        cn_solver = splu(left)
        for _ in range(steps - 1):
            state = cn_solver.solve(right @ state)
        solves += steps - 1

    return AdvanceResult(values=state, linear_solves=solves)


def advance(
    method: str,
    matrix: csc_matrix,
    initial: np.ndarray,
    dt: float,
    steps: int,
) -> AdvanceResult:
    if method == "backward_euler":
        return advance_backward_euler(matrix, initial, dt, steps)
    if method == "crank_nicolson":
        return advance_crank_nicolson(matrix, initial, dt, steps)
    if method == "rannacher_r1":
        return advance_rannacher_r1(matrix, initial, dt, steps)
    raise ValueError(f"unknown time integrator: {method}")



def with_zero_boundaries(grid: HeatGrid, interior: np.ndarray) -> np.ndarray:
    """Return [0, U_1, ..., U_{N-1}, 0] for diagnostics/output."""
    values = np.asarray(interior, dtype=float)
    if values.shape != (grid.N - 1,):
        raise ValueError("interior state shape does not match grid")
    return np.concatenate(([0.0], values, [0.0]))


def total_variation(grid: HeatGrid, interior: np.ndarray) -> float:
    full = with_zero_boundaries(grid, interior)
    return float(np.sum(np.abs(np.diff(full))))


def monotonicity_violations(
    grid: HeatGrid,
    interior: np.ndarray,
    *,
    tolerance: float,
) -> list[dict[str, float | int | str]]:
    """Find slopes inconsistent with the symmetric diffused-box shape.

    On [0, 1/2] the exact/semidiscrete solution should be nondecreasing;
    on [1/2, 1] it should be nonincreasing.  This parameter-free half-domain
    check captures the jump-localized zigzags without selecting a visual crop.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    full = with_zero_boundaries(grid, interior)
    x_full = np.linspace(0.0, 1.0, grid.N + 1)
    violations: list[dict[str, float | int | str]] = []

    for edge in range(grid.N):
        delta = float(full[edge + 1] - full[edge])
        midpoint = 0.5 * float(x_full[edge] + x_full[edge + 1])

        if midpoint < 0.5 and delta < -tolerance:
            violations.append(
                {
                    "side": "left",
                    "edge_left_index": edge,
                    "edge_right_index": edge + 1,
                    "x_left": float(x_full[edge]),
                    "x_right": float(x_full[edge + 1]),
                    "delta": delta,
                }
            )
        elif midpoint > 0.5 and delta > tolerance:
            violations.append(
                {
                    "side": "right",
                    "edge_left_index": edge,
                    "edge_right_index": edge + 1,
                    "x_left": float(x_full[edge]),
                    "x_right": float(x_full[edge + 1]),
                    "delta": delta,
                }
            )

    return violations
