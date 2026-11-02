"""Tutorial-007: transparent uniform-node anisotropic diffusion and face fluxes.

For each *interior vertex* a square dual control volume has four faces of
length h. The interior rows of the assembled operator are the algebraic
reduction of the shared-face flux divergence; no boundary ghost values exist.
This is NOT a general unstructured-mesh MPFA implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


Method = Literal["M0", "M1"]
Field = Literal["sinesine", "affine", "xy"]


def tensor(theta_deg: float, principal: tuple[float, float] = (10.0, 1.0)
           ) -> tuple[float, float, float]:
    if not math.isfinite(theta_deg):
        raise ValueError("rotation must be finite and in degrees")
    a, b = principal
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("principal conductivities must be finite positive")
    t = math.radians(theta_deg)
    co, si = math.cos(t), math.sin(t)
    return a * co * co + b * si * si, (a - b) * co * si, a * si * si + b * co * co


def exact_and_source(field: Field, x: np.ndarray, y: np.ndarray,
                     kxx: float, kxy: float, kyy: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    if field == "sinesine":
        sx, sy = np.sin(np.pi * x), np.sin(np.pi * y)
        cx, cy = np.cos(np.pi * x), np.cos(np.pi * y)
        analytic_u = sx * sy
        source = np.pi ** 2 * ((kxx + kyy) * analytic_u - 2.0 * kxy * cx * cy)
        # Dirichlet T=0 is mathematically exact on all four faces.
        # sin(pi) is merely about 1e-16 in finite precision, not exact zero.
        u = np.where((x <= 0) | (x >= 1) | (y <= 0) | (y >= 1),
                     0.0, analytic_u)
        return u, source
    if field == "affine":
        return np.broadcast_arrays(x, np.zeros_like(x + y))
    if field == "xy":
        return np.broadcast_arrays(x * y, np.full_like(x + y, -2.0 * kxy))
    raise ValueError("unrecognized manufactured field")


def analytic_face_flux(field: Field, theta_deg: float, n: int,
                       principal: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    kxx, kxy, kyy = tensor(theta_deg, principal)
    h = 1.0 / n
    # The vertical face grid is (i+1/2,j), i=0..N-1, j=1..N-1.
    xv, yv = np.meshgrid((np.arange(n) + 0.5) * h,
                         np.arange(1, n) * h, indexing="ij")
    # The horizontal face grid is (i,j+1/2), i=1..N-1, j=0..N-1.
    xh, yh = np.meshgrid(np.arange(1, n) * h,
                         (np.arange(n) + 0.5) * h, indexing="ij")
    def gradient(x: np.ndarray, y: np.ndarray):
        if field == "sinesine":
            return (np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                    np.pi * np.sin(np.pi * x) * np.cos(np.pi * y))
        if field == "affine":
            return np.ones_like(x), np.zeros_like(x)
        if field == "xy":
            return y, x
        raise ValueError("invalid field")
    gx, gy = gradient(xv, yv)
    qx = -kxx * gx - kxy * gy
    gx, gy = gradient(xh, yh)
    qy = -kxy * gx - kyy * gy
    return qx, qy


@dataclass(frozen=True)
class System:
    A: csr_matrix
    b: np.ndarray
    k: tuple[float, float, float]
    n: int
    method: str
    field: str


def _index(i: int, j: int, n: int) -> int:
    return (i - 1) * (n - 1) + (j - 1)


def assemble(n: int, theta_deg: float, method: Method, field: Field,
             principal: tuple[float, float] = (10.0, 1.0)) -> System:
    if isinstance(n, bool) or not isinstance(n, int) or n < 4:
        raise ValueError("N must be an integer >= 4")
    if method not in ("M0", "M1"):
        raise ValueError("method must be M0 or M1")
    kxx, kxy, kyy = tensor(theta_deg, principal)
    h = 1.0 / n
    w = 1.0 / (h * h)
    # These nine coefficients are exactly the divergence of the two shared
    # flux formulas below, divided by the dual volume h^2.
    offsets = [(0, 0, 2.0 * (kxx + kyy) * w),
               (1, 0, -kxx * w), (-1, 0, -kxx * w),
               (0, 1, -kyy * w), (0, -1, -kyy * w)]
    if method == "M1" and kxy != 0.0:
        offsets += [(1, 1, -kxy * w / 2), (-1, -1, -kxy * w / 2),
                    (1, -1, +kxy * w / 2), (-1, 1, +kxy * w / 2)]
    size = (n - 1) ** 2
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(size, dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            row = _index(i, j, n)
            _, source = exact_and_source(field, np.array(i * h), np.array(j * h),
                                         kxx, kxy, kyy)
            value = float(source)
            for di, dj, coeff in offsets:
                ii, jj = i + di, j + dj
                if ii in (0, n) or jj in (0, n):
                    boundary, _ = exact_and_source(
                        field, np.array(ii * h), np.array(jj * h), kxx, kxy, kyy)
                    value -= coeff * float(boundary)
                else:
                    rows.append(row)
                    cols.append(_index(ii, jj, n))
                    values.append(coeff)
            rhs[row] = value
    matrix = csr_matrix((np.asarray(values, dtype=float), (rows, cols)),
                        shape=(size, size))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(matrix.data)):
        raise FloatingPointError("assembly contains nonfinite numbers")
    return System(matrix, rhs, (kxx, kxy, kyy), n, method, field)


def dual_face_flux(u: np.ndarray, theta_deg: float, method: Method,
                   principal: tuple[float, float] = (10.0, 1.0)
                   ) -> tuple[np.ndarray, np.ndarray]:
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] < 5:
        raise ValueError("full square nodal array required")
    if method not in ("M0", "M1"):
        raise ValueError("invalid method")
    n = u.shape[0] - 1
    h = 1.0 / n
    kxx, kxy, kyy = tensor(theta_deg, principal)
    qx = -kxx * (u[1:, 1:n] - u[:-1, 1:n]) / h
    qy = -kyy * (u[1:n, 1:] - u[1:n, :-1]) / h
    if method == "M1":
        qx = qx - (kxy / (4*h)) * (
            u[:-1, 2:] - u[:-1, :-2] + u[1:, 2:] - u[1:, :-2])
        qy = qy - (kxy / (4*h)) * (
            u[2:, :-1] - u[:-2, :-1] + u[2:, 1:] - u[:-2, 1:])
    if qx.shape != (n, n - 1) or qy.shape != (n - 1, n):
        raise AssertionError("incorrect dual face dimensions")
    return qx, qy


def nodal_solution(system: System) -> tuple[np.ndarray, float]:
    n = system.n
    h = 1.0 / n
    xx, yy = np.meshgrid(np.arange(n + 1) * h, np.arange(n + 1) * h,
                         indexing="ij")
    kxx, kxy, kyy = system.k
    boundary, _ = exact_and_source(system.field, xx, yy, kxx, kxy, kyy)
    u = np.array(boundary, dtype=np.float64, copy=True)
    internal = np.asarray(spsolve(system.A, system.b), dtype=np.float64)
    if internal.size != (n-1)**2 or not np.all(np.isfinite(internal)):
        raise RuntimeError("nonfinite or incorrect sparse solve")
    residual = system.A @ internal - system.b
    residual_relative = float(np.max(np.abs(residual)) /
                              max(1.0, float(np.max(np.abs(system.b)))))
    u[1:n, 1:n] = internal.reshape((n - 1, n - 1))
    return u, residual_relative


def report_metrics(u: np.ndarray, theta_deg: float, method: Method, field: Field,
                   principal: tuple[float, float] = (10.0, 1.0)) -> dict[str, float]:
    n = u.shape[0] - 1
    h = 1.0 / n
    kxx, kxy, kyy = tensor(theta_deg, principal)
    xx, yy = np.meshgrid(np.arange(n + 1) * h, np.arange(n + 1) * h,
                         indexing="ij")
    reference, source = exact_and_source(field, xx, yy, kxx, kxy, kyy)
    err = u[1:n, 1:n] - reference[1:n, 1:n]
    qx, qy = dual_face_flux(u, theta_deg, method, principal)
    ex, ey = analytic_face_flux(field, theta_deg, n, principal)
    # Vertex-centered dual control volumes V_ij tile only Omega_h.
    balance = h * (qx[1:, :] - qx[:-1, :] +
                   qy[:, 1:] - qy[:, :-1]) - h*h*source[1:n, 1:n]
    flux_scale = max(1., float(np.max(np.abs(h*qx))), float(np.max(np.abs(h*qy))),
                     h*h*float(np.max(np.abs(source[1:n, 1:n]))))
    net_dual_flux = h * (np.sum(qx[-1, :]) - np.sum(qx[0, :]) +
                         np.sum(qy[:, -1]) - np.sum(qy[:, 0]))
    budget = float(net_dual_flux - h*h*np.sum(source[1:n, 1:n]))
    return {
        "T_error_l2": float(np.sqrt(h*h*np.sum(err*err))),
        "T_error_linf": float(np.max(np.abs(err))),
        "q_x_error_linf": float(np.max(np.abs(qx-ex))),
        "q_y_error_linf": float(np.max(np.abs(qy-ey))),
        "dual_cell_balance_linf": float(np.max(np.abs(balance))),
        "dual_cell_balance_scaled": float(np.max(np.abs(balance))/flux_scale),
        "dual_boundary_budget_defect": budget,
        "dual_boundary_budget_scaled": abs(budget)/flux_scale,
    }
