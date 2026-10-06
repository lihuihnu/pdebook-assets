"""Minimal P1 finite-element core for tutorial-002.

The core is intentionally limited to the homogeneous Laplace equation on the
frozen L-shaped domain.  It exposes the geometry, P1 stiffness assembly and
Dirichlet solve explicitly; SciPy is used only for sparse storage/solve.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray
    triangles: np.ndarray
    boundary_nodes: np.ndarray
    boundary_edges: np.ndarray
    N: int
    beta: float


@dataclass(frozen=True)
class AssemblyAudit:
    min_signed_area: float
    max_local_symmetry_defect: float
    max_local_row_sum_defect: float


@dataclass(frozen=True)
class SolveResult:
    mesh: Mesh
    values: np.ndarray
    stiffness: csr_matrix
    free_nodes: np.ndarray
    boundary_nodes: np.ndarray
    reduced_residual_l2: float
    free_block_symmetry_defect: float
    assembly_audit: AssemblyAudit


def _graded_coordinate(x: float, y: float, beta: float) -> tuple[float, float]:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    rho = max(abs(x), abs(y))
    if rho == 0.0 or beta == 1.0:
        return x, y
    factor = rho ** (beta - 1.0)
    return factor * x, factor * y


def build_l_shape_mesh(N: int, beta: float = 1.0) -> Mesh:
    """Build the frozen alternating-diagonal L-shaped triangulation.

    Start from an N-by-N Cartesian partition of [-1,1]^2, remove cells in
    [0,1] x [-1,0], then optionally apply the frozen L-infinity radial grading
    map Phi_beta to vertices.  Unused vertices are pruned deterministically.
    """
    if N < 2 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 2")

    coords = np.linspace(-1.0, 1.0, N + 1)
    raw_points: list[tuple[float, float]] = []
    raw_index: dict[tuple[int, int], int] = {}

    for j, y in enumerate(coords):
        for i, x in enumerate(coords):
            raw_index[(i, j)] = len(raw_points)
            raw_points.append(_graded_coordinate(float(x), float(y), beta))

    raw_triangles: list[tuple[int, int, int]] = []
    for j in range(N):
        y1 = float(coords[j + 1])
        for i in range(N):
            x0 = float(coords[i])

            # Axes are grid lines because N is even.  Remove exactly the cells
            # occupying the deleted fourth-quadrant square.
            if x0 >= 0.0 and y1 <= 0.0:
                continue

            bl = raw_index[(i, j)]
            br = raw_index[(i + 1, j)]
            tr = raw_index[(i + 1, j + 1)]
            tl = raw_index[(i, j + 1)]

            if (i + j) % 2 == 0:
                raw_triangles.append((bl, br, tr))
                raw_triangles.append((bl, tr, tl))
            else:
                raw_triangles.append((bl, br, tl))
                raw_triangles.append((br, tr, tl))

    used = sorted({node for tri in raw_triangles for node in tri})
    remap = {old: new for new, old in enumerate(used)}

    points = np.asarray([raw_points[node] for node in used], dtype=float)
    triangles = np.asarray(
        [[remap[node] for node in tri] for tri in raw_triangles],
        dtype=np.int64,
    )

    edge_counts: Counter[tuple[int, int]] = Counter()
    for tri in triangles:
        a, b, c = (int(v) for v in tri)
        for edge in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted(edge))] += 1

    boundary_edges = np.asarray(
        sorted(edge for edge, count in edge_counts.items() if count == 1),
        dtype=np.int64,
    )
    boundary_nodes = np.asarray(
        sorted({node for edge in boundary_edges for node in edge}),
        dtype=np.int64,
    )

    return Mesh(
        points=points,
        triangles=triangles,
        boundary_nodes=boundary_nodes,
        boundary_edges=boundary_edges,
        N=N,
        beta=float(beta),
    )


def triangle_geometry(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """Return signed area and gradients of the three P1 basis functions."""
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2)")

    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]

    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = 0.5 * float(det)
    if not area > 0.0:
        raise ValueError(f"triangle has non-positive signed area: {area}")

    grad_x = np.asarray([y1 - y2, y2 - y0, y0 - y1], dtype=float) / det
    grad_y = np.asarray([x2 - x1, x0 - x2, x1 - x0], dtype=float) / det
    gradients = np.column_stack((grad_x, grad_y))
    return area, gradients


def local_stiffness(vertices: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    area, gradients = triangle_geometry(vertices)
    matrix = area * (gradients @ gradients.T)
    return matrix, area, gradients


def assemble_stiffness(mesh: Mesh) -> tuple[csr_matrix, AssemblyAudit]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    min_area = math.inf
    max_symmetry = 0.0
    max_row_sum = 0.0

    for tri in mesh.triangles:
        local, area, _ = local_stiffness(mesh.points[tri])
        min_area = min(min_area, area)
        max_symmetry = max(
            max_symmetry,
            float(np.max(np.abs(local - local.T))),
        )
        max_row_sum = max(
            max_row_sum,
            float(np.max(np.abs(np.sum(local, axis=1)))),
        )

        for a in range(3):
            for b in range(3):
                rows.append(int(tri[a]))
                cols.append(int(tri[b]))
                data.append(float(local[a, b]))

    n = mesh.points.shape[0]
    matrix = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    audit = AssemblyAudit(
        min_signed_area=float(min_area),
        max_local_symmetry_defect=max_symmetry,
        max_local_row_sum_defect=max_row_sum,
    )
    return matrix, audit


def solve_dirichlet_laplace(
    mesh: Mesh,
    boundary_value,
) -> SolveResult:
    """Solve -Delta u = 0 with exact nodal Dirichlet values on all boundaries."""
    stiffness, audit = assemble_stiffness(mesh)

    is_free = np.ones(mesh.points.shape[0], dtype=bool)
    is_free[mesh.boundary_nodes] = False
    free_nodes = np.flatnonzero(is_free)

    boundary_values = np.asarray(
        boundary_value(
            mesh.points[mesh.boundary_nodes, 0],
            mesh.points[mesh.boundary_nodes, 1],
        ),
        dtype=float,
    )

    a_ff = stiffness[free_nodes][:, free_nodes].tocsr()
    a_fb = stiffness[free_nodes][:, mesh.boundary_nodes].tocsr()
    rhs = -(a_fb @ boundary_values)

    free_values = spsolve(a_ff, rhs)
    values = np.empty(mesh.points.shape[0], dtype=float)
    values[mesh.boundary_nodes] = boundary_values
    values[free_nodes] = free_values

    residual = a_ff @ free_values - rhs
    residual_l2 = float(
        np.linalg.norm(residual) / max(float(np.linalg.norm(rhs)), 1.0)
    )

    asymmetry = a_ff - a_ff.T
    symmetry_defect = (
        0.0
        if asymmetry.nnz == 0
        else float(np.max(np.abs(asymmetry.data)))
    )

    return SolveResult(
        mesh=mesh,
        values=values,
        stiffness=stiffness,
        free_nodes=free_nodes,
        boundary_nodes=mesh.boundary_nodes,
        reduced_residual_l2=residual_l2,
        free_block_symmetry_defect=symmetry_defect,
        assembly_audit=audit,
    )


def mesh_quality(mesh: Mesh) -> dict[str, float]:
    """Return simple triangle-quality diagnostics for later matched-DOF use."""
    min_angle = math.inf
    max_edge_ratio = 0.0

    for tri in mesh.triangles:
        p = mesh.points[tri]
        lengths = np.asarray(
            [
                np.linalg.norm(p[1] - p[0]),
                np.linalg.norm(p[2] - p[1]),
                np.linalg.norm(p[0] - p[2]),
            ],
            dtype=float,
        )
        max_edge_ratio = max(
            max_edge_ratio,
            float(np.max(lengths) / np.min(lengths)),
        )

        a, b, c = lengths
        angles = [
            math.acos(np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0)),
            math.acos(np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)),
            math.acos(np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)),
        ]
        min_angle = min(min_angle, *(float(v) for v in angles))

    return {
        "min_angle_deg": math.degrees(min_angle),
        "max_edge_ratio": max_edge_ratio,
    }
