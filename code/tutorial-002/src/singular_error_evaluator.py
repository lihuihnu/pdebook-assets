"""Corner-aware error integration for the ET002-03 singular benchmark.

The verified generic evaluator remains unchanged.  Away from the re-entrant
corner this module delegates to it.  For triangles incident to the origin, a
Duffy map followed by s=z^3 regularizes the r^{-1/3} gradient singularity and
a tensor Gauss-Legendre rule integrates the resulting smooth integrand.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from error_evaluator import ErrorResult, evaluate_error


def _signed_area(vertices: np.ndarray) -> float:
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    return 0.5 * float((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _p1_gradient(vertices: np.ndarray, values: np.ndarray) -> np.ndarray:
    area = _signed_area(vertices)
    if not area > 0.0:
        raise ValueError("corner-aware evaluator received a non-positive triangle")
    det = 2.0 * area
    grad_x = np.asarray(
        [
            vertices[1, 1] - vertices[2, 1],
            vertices[2, 1] - vertices[0, 1],
            vertices[0, 1] - vertices[1, 1],
        ],
        dtype=float,
    ) / det
    grad_y = np.asarray(
        [
            vertices[2, 0] - vertices[1, 0],
            vertices[0, 0] - vertices[2, 0],
            vertices[1, 0] - vertices[0, 0],
        ],
        dtype=float,
    ) / det
    gradients = np.column_stack((grad_x, grad_y))
    return np.asarray(values, dtype=float) @ gradients


def _unit_interval_gauss(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2:
        raise ValueError("Gauss order must be at least 2")
    nodes, weights = leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _corner_triangle_integrals(
    vertices: np.ndarray,
    values: np.ndarray,
    exact_value,
    exact_gradient,
    *,
    gauss_order: int,
    origin_tolerance: float,
) -> tuple[float, float, float, float]:
    radii = np.linalg.norm(vertices, axis=1)
    origin_candidates = np.flatnonzero(radii <= origin_tolerance)
    if origin_candidates.size != 1:
        raise ValueError("corner triangle must contain exactly one origin vertex")

    origin_local = int(origin_candidates[0])
    other = [index for index in range(3) if index != origin_local]

    o = vertices[origin_local]
    a = vertices[other[0]]
    b = vertices[other[1]]
    u0 = float(values[origin_local])
    ua = float(values[other[0]])
    ub = float(values[other[1]])

    a_rel = a - o
    b_rel = b - o
    det = abs(
        float(a_rel[0] * b_rel[1] - a_rel[1] * b_rel[0])
    )
    if not det > 0.0:
        raise ValueError("degenerate corner triangle")

    z_nodes, z_weights = _unit_interval_gauss(gauss_order)
    t_nodes, t_weights = _unit_interval_gauss(gauss_order)

    z, t = np.meshgrid(z_nodes, t_nodes, indexing="ij")
    wz, wt = np.meshgrid(z_weights, t_weights, indexing="ij")

    s = z**3
    ray = (1.0 - t)[..., None] * a_rel + t[..., None] * b_rel
    points = o + s[..., None] * ray
    flat_points = points.reshape((-1, 2))

    uh = (
        (1.0 - s) * u0
        + s * (1.0 - t) * ua
        + s * t * ub
    ).reshape(-1)

    # Duffy map: x=s((1-t)a+t b), dA=det*s ds dt.
    # Radial regularization s=z^3 gives ds=3 z^2 dz, hence
    # dA=3*det*z^5 dz dt.
    weights = (wz * wt * (3.0 * det * z**5)).reshape(-1)

    ue = np.asarray(
        exact_value(flat_points[:, 0], flat_points[:, 1]),
        dtype=float,
    )
    grad_exact = np.asarray(
        exact_gradient(flat_points[:, 0], flat_points[:, 1]),
        dtype=float,
    )
    if grad_exact.shape != (flat_points.shape[0], 2):
        raise ValueError("exact_gradient must return shape (n, 2)")

    grad_uh = _p1_gradient(vertices, values)
    value_diff = ue - uh
    grad_diff = grad_exact - grad_uh

    l2_sq = float(np.sum(weights * value_diff * value_diff))
    h1_sq = float(np.sum(weights * np.sum(grad_diff * grad_diff, axis=1)))
    exact_l2_sq = float(np.sum(weights * ue * ue))
    exact_h1_sq = float(
        np.sum(weights * np.sum(grad_exact * grad_exact, axis=1))
    )
    return l2_sq, h1_sq, exact_l2_sq, exact_h1_sq


def evaluate_singular_error(
    points: np.ndarray,
    triangles: np.ndarray,
    nodal_values: np.ndarray,
    exact_value,
    exact_gradient,
    *,
    noncorner_subdivision_levels: int = 2,
    corner_gauss_order: int = 24,
    origin_tolerance: float = 1.0e-14,
    return_element_h1_sq: bool = False,
) -> ErrorResult:
    triangles = np.asarray(triangles, dtype=np.int64)
    points = np.asarray(points, dtype=float)
    nodal_values = np.asarray(nodal_values, dtype=float)

    corner_mask = np.zeros(triangles.shape[0], dtype=bool)
    for index, tri in enumerate(triangles):
        corner_mask[index] = bool(
            np.any(np.linalg.norm(points[tri], axis=1) <= origin_tolerance)
        )

    noncorner_indices = np.flatnonzero(~corner_mask)
    corner_indices = np.flatnonzero(corner_mask)
    if corner_indices.size == 0:
        raise AssertionError("singular benchmark mesh has no origin-adjacent triangles")

    element_h1_sq = (
        np.zeros(triangles.shape[0], dtype=float)
        if return_element_h1_sq
        else None
    )

    total_l2_sq = 0.0
    total_h1_sq = 0.0
    total_exact_l2_sq = 0.0
    total_exact_h1_sq = 0.0

    if noncorner_indices.size:
        noncorner = evaluate_error(
            points,
            triangles[noncorner_indices],
            nodal_values,
            exact_value,
            exact_gradient,
            subdivision_levels=noncorner_subdivision_levels,
            return_element_h1_sq=return_element_h1_sq,
        )
        total_l2_sq += noncorner.l2**2
        total_h1_sq += noncorner.h1_seminorm**2
        total_exact_l2_sq += noncorner.exact_l2**2
        total_exact_h1_sq += noncorner.exact_h1_seminorm**2
        if return_element_h1_sq:
            assert noncorner.element_h1_sq is not None
            element_h1_sq[noncorner_indices] = noncorner.element_h1_sq

    for triangle_index in corner_indices:
        tri = triangles[triangle_index]
        values = nodal_values[tri]
        l2_sq, h1_sq, exact_l2_sq, exact_h1_sq = _corner_triangle_integrals(
            points[tri],
            values,
            exact_value,
            exact_gradient,
            gauss_order=corner_gauss_order,
            origin_tolerance=origin_tolerance,
        )
        total_l2_sq += l2_sq
        total_h1_sq += h1_sq
        total_exact_l2_sq += exact_l2_sq
        total_exact_h1_sq += exact_h1_sq
        if return_element_h1_sq:
            element_h1_sq[triangle_index] = h1_sq

    l2 = math.sqrt(total_l2_sq)
    h1 = math.sqrt(total_h1_sq)
    exact_l2 = math.sqrt(total_exact_l2_sq)
    exact_h1 = math.sqrt(total_exact_h1_sq)

    return ErrorResult(
        l2=l2,
        h1_seminorm=h1,
        relative_l2=l2 / exact_l2,
        relative_h1_seminorm=h1 / exact_h1,
        exact_l2=exact_l2,
        exact_h1_seminorm=exact_h1,
        element_h1_sq=element_h1_sq,
    )
