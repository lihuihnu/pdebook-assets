"""Independent error evaluator for tutorial-002.

This module deliberately does not import fem_core.  It receives a mesh and a
nodal P1 solution as plain arrays, performs its own virtual refinement, and
integrates exact-vs-discrete errors with a fixed seven-point Dunavant rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# Seven-point degree-five Dunavant rule in barycentric coordinates.
_DUNAVANT_LAMBDA = np.asarray(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=float,
)
_DUNAVANT_WEIGHT = np.asarray(
    [
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ErrorResult:
    l2: float
    h1_seminorm: float
    relative_l2: float
    relative_h1_seminorm: float
    exact_l2: float
    exact_h1_seminorm: float
    element_h1_sq: np.ndarray | None


def _signed_area(vertices: np.ndarray) -> float:
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    return 0.5 * float((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _p1_gradient(vertices: np.ndarray, values: np.ndarray) -> np.ndarray:
    area = _signed_area(vertices)
    if not area > 0.0:
        raise ValueError("error evaluator received a non-positive triangle")
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


def _subdivide_once(vertices: np.ndarray) -> list[np.ndarray]:
    a, b, c = vertices
    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c)
    ca = 0.5 * (c + a)
    return [
        np.asarray([a, ab, ca], dtype=float),
        np.asarray([ab, b, bc], dtype=float),
        np.asarray([ca, bc, c], dtype=float),
        np.asarray([ab, bc, ca], dtype=float),
    ]


def _virtual_subtriangles(vertices: np.ndarray, levels: int) -> list[np.ndarray]:
    if levels < 0:
        raise ValueError("subdivision levels must be non-negative")
    pieces = [np.asarray(vertices, dtype=float)]
    for _ in range(levels):
        next_pieces: list[np.ndarray] = []
        for piece in pieces:
            next_pieces.extend(_subdivide_once(piece))
        pieces = next_pieces
    return pieces


def _barycentric_coordinates(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        [
            [vertices[0, 0], vertices[1, 0], vertices[2, 0]],
            [vertices[0, 1], vertices[1, 1], vertices[2, 1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    rhs = np.vstack(
        (
            points[:, 0],
            points[:, 1],
            np.ones(points.shape[0], dtype=float),
        )
    )
    return np.linalg.solve(matrix, rhs).T


def evaluate_error(
    points: np.ndarray,
    triangles: np.ndarray,
    nodal_values: np.ndarray,
    exact_value,
    exact_gradient,
    *,
    subdivision_levels: int = 2,
    return_element_h1_sq: bool = False,
) -> ErrorResult:
    l2_sq = 0.0
    h1_sq = 0.0
    exact_l2_sq = 0.0
    exact_h1_sq = 0.0
    local_h1: list[float] = []

    for tri in np.asarray(triangles, dtype=np.int64):
        vertices = np.asarray(points[tri], dtype=float)
        values = np.asarray(nodal_values[tri], dtype=float)
        grad_uh = _p1_gradient(vertices, values)

        element_l2_sq = 0.0
        element_h1_sq = 0.0

        for subtriangle in _virtual_subtriangles(vertices, subdivision_levels):
            area = _signed_area(subtriangle)
            if not area > 0.0:
                raise ValueError("virtual refinement produced a non-positive triangle")

            quadrature_points = _DUNAVANT_LAMBDA @ subtriangle
            bary = _barycentric_coordinates(quadrature_points, vertices)
            uh = bary @ values

            ue = np.asarray(
                exact_value(
                    quadrature_points[:, 0],
                    quadrature_points[:, 1],
                ),
                dtype=float,
            )
            grad_exact = np.asarray(
                exact_gradient(
                    quadrature_points[:, 0],
                    quadrature_points[:, 1],
                ),
                dtype=float,
            )
            if grad_exact.shape != (quadrature_points.shape[0], 2):
                raise ValueError("exact_gradient must return shape (n, 2)")

            value_diff = ue - uh
            grad_diff = grad_exact - grad_uh

            element_l2_sq += area * float(
                np.sum(_DUNAVANT_WEIGHT * value_diff * value_diff)
            )
            element_h1_sq += area * float(
                np.sum(
                    _DUNAVANT_WEIGHT
                    * np.sum(grad_diff * grad_diff, axis=1)
                )
            )
            exact_l2_sq += area * float(
                np.sum(_DUNAVANT_WEIGHT * ue * ue)
            )
            exact_h1_sq += area * float(
                np.sum(
                    _DUNAVANT_WEIGHT
                    * np.sum(grad_exact * grad_exact, axis=1)
                )
            )

        l2_sq += element_l2_sq
        h1_sq += element_h1_sq
        if return_element_h1_sq:
            local_h1.append(element_h1_sq)

    l2 = math.sqrt(l2_sq)
    h1 = math.sqrt(h1_sq)
    exact_l2 = math.sqrt(exact_l2_sq)
    exact_h1 = math.sqrt(exact_h1_sq)

    return ErrorResult(
        l2=l2,
        h1_seminorm=h1,
        relative_l2=l2 / exact_l2,
        relative_h1_seminorm=h1 / exact_h1,
        exact_l2=exact_l2,
        exact_h1_seminorm=exact_h1,
        element_h1_sq=(
            np.asarray(local_h1, dtype=float)
            if return_element_h1_sq
            else None
        ),
    )


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)
