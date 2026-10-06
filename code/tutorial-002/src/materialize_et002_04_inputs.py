#!/usr/bin/env python3
"""Materialize frozen U128 sidecars required by ET002-04.

This script is a deterministic replay of already-verified ET002-02/03 cases.
It does not define a new experiment.  Sidecars are written only after identity
checks against the authoritative convergence data pass.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from fem_core import build_l_shape_mesh, solve_dirichlet_laplace, triangle_geometry
from run_et002_02 import smooth_gradient, smooth_value
from run_et002_03 import singular_gradient, singular_value
from singular_error_evaluator import evaluate_singular_error


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
ET02_CSV = RESULT_ROOT / "et002-02/convergence.csv"
ET03_CSV = RESULT_ROOT / "et002-03/convergence.csv"
ET03_ELEMENT = RESULT_ROOT / "et002-03/element_h1_sq.csv"
OUT = RESULT_ROOT / "et002-04-inputs"

N = 128
METRIC_REL_TOL = 1.0e-12
RESIDUAL_ABS_TOL = 1.0e-13
ELEMENT_REL_TOL = 1.0e-12
ELEMENT_ABS_TOL = 1.0e-18


def read_row(path: Path, target_n: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["N"]) == target_n]
    if len(rows) != 1:
        raise AssertionError(f"{path}: expected exactly one N={target_n} row")
    return rows[0]


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_singular_elements() -> np.ndarray:
    values: list[tuple[int, float]] = []
    with ET03_ELEMENT.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["N"]) == N:
                values.append((int(row["triangle_id"]), float(row["h1_error_sq"])))
    if not values:
        raise AssertionError("no frozen N=128 singular element contributions found")
    values.sort(key=lambda item: item[0])
    ids = [item[0] for item in values]
    if ids != list(range(len(values))):
        raise AssertionError("frozen singular triangle ids are not contiguous")
    return np.asarray([item[1] for item in values], dtype=float)


def main() -> None:
    mesh = build_l_shape_mesh(N, 1.0)
    if mesh.points.shape[0] != 12545 or mesh.triangles.shape[0] != 24576:
        raise AssertionError("U128 mesh inventory drifted")

    cases = [
        ("smooth", smooth_value, smooth_gradient, ET02_CSV),
        ("singular", singular_value, singular_gradient, ET03_CSV),
    ]

    materialized: dict[str, dict[str, object]] = {}
    check_rows: list[dict[str, object]] = []

    for name, exact_value, exact_gradient, convergence_path in cases:
        authoritative = read_row(convergence_path, N)
        result = solve_dirichlet_laplace(mesh, exact_value)
        error = evaluate_singular_error(
            mesh.points,
            mesh.triangles,
            result.values,
            exact_value,
            exact_gradient,
            noncorner_subdivision_levels=2,
            corner_gauss_order=24,
            return_element_h1_sq=True,
        )
        if error.element_h1_sq is None:
            raise AssertionError(f"{name}: missing element H1 contributions")

        target_h1 = float(authoritative["h1_error"])
        target_l2 = float(authoritative["l2_error"])
        target_residual = float(authoritative["reduced_residual_l2"])

        h1_rel = relative_difference(error.h1_seminorm, target_h1)
        l2_rel = relative_difference(error.l2, target_l2)
        residual_abs = abs(result.reduced_residual_l2 - target_residual)

        if h1_rel > METRIC_REL_TOL:
            raise AssertionError(f"{name}: H1 identity drift {h1_rel}")
        if l2_rel > METRIC_REL_TOL:
            raise AssertionError(f"{name}: L2 identity drift {l2_rel}")
        if residual_abs > RESIDUAL_ABS_TOL:
            raise AssertionError(f"{name}: residual identity drift {residual_abs}")

        materialized[name] = {
            "values": result.values,
            "exact_value": exact_value,
            "element_h1_sq": error.element_h1_sq,
            "h1_error": error.h1_seminorm,
            "l2_error": error.l2,
            "residual": result.reduced_residual_l2,
        }
        check_rows.append(
            {
                "problem": name,
                "materialized_h1_error": error.h1_seminorm,
                "authoritative_h1_error": target_h1,
                "h1_relative_difference": h1_rel,
                "materialized_l2_error": error.l2,
                "authoritative_l2_error": target_l2,
                "l2_relative_difference": l2_rel,
                "materialized_residual": result.reduced_residual_l2,
                "authoritative_residual": target_residual,
                "residual_absolute_difference": residual_abs,
            }
        )

    frozen_singular = load_frozen_singular_elements()
    singular_current = np.asarray(materialized["singular"]["element_h1_sq"], dtype=float)
    if frozen_singular.shape != singular_current.shape:
        raise AssertionError("singular frozen/materialized element count mismatch")

    absolute = np.abs(singular_current - frozen_singular)
    relative = absolute / np.maximum(np.abs(frozen_singular), 1.0e-300)
    bad = (absolute > ELEMENT_ABS_TOL) & (relative > ELEMENT_REL_TOL)
    if np.any(bad):
        index = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            "singular element identity drift at triangle "
            f"{index}: current={singular_current[index]:.16e}, "
            f"frozen={frozen_singular[index]:.16e}, "
            f"abs={absolute[index]:.3e}, rel={relative[index]:.3e}"
        )

    check_rows[1]["max_singular_element_absolute_difference"] = float(np.max(absolute))
    check_rows[1]["max_singular_element_relative_difference"] = float(np.max(relative))
    check_rows[0]["max_singular_element_absolute_difference"] = ""
    check_rows[0]["max_singular_element_relative_difference"] = ""

    OUT.mkdir(parents=True, exist_ok=True)

    node_rows: list[dict[str, object]] = []
    smooth_solution: list[dict[str, object]] = []
    singular_solution: list[dict[str, object]] = []
    for node_id, (x, y) in enumerate(mesh.points):
        node_rows.append({"node_id": node_id, "x": float(x), "y": float(y)})
        smooth_solution.append(
            {
                "node_id": node_id,
                "u_h": float(materialized["smooth"]["values"][node_id]),
                "u_exact": float(smooth_value(np.asarray([x]), np.asarray([y]))[0]),
            }
        )
        singular_solution.append(
            {
                "node_id": node_id,
                "u_h": float(materialized["singular"]["values"][node_id]),
                "u_exact": float(singular_value(np.asarray([x]), np.asarray([y]))[0]),
            }
        )

    triangle_rows: list[dict[str, object]] = []
    for triangle_id, tri in enumerate(mesh.triangles):
        vertices = mesh.points[tri]
        area, _ = triangle_geometry(vertices)
        centroid = np.mean(vertices, axis=0)
        triangle_rows.append(
            {
                "triangle_id": triangle_id,
                "node0": int(tri[0]),
                "node1": int(tri[1]),
                "node2": int(tri[2]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_r": float(np.hypot(centroid[0], centroid[1])),
                "area": float(area),
            }
        )

    write_csv(OUT / "u128_nodes.csv", ["node_id", "x", "y"], node_rows)
    write_csv(
        OUT / "u128_triangles.csv",
        [
            "triangle_id",
            "node0",
            "node1",
            "node2",
            "centroid_x",
            "centroid_y",
            "centroid_r",
            "area",
        ],
        triangle_rows,
    )
    write_csv(
        OUT / "smooth_u128_solution.csv",
        ["node_id", "u_h", "u_exact"],
        smooth_solution,
    )
    write_csv(
        OUT / "singular_u128_solution.csv",
        ["node_id", "u_h", "u_exact"],
        singular_solution,
    )
    write_csv(
        OUT / "smooth_u128_element_h1_sq.csv",
        ["triangle_id", "h1_error_sq"],
        [
            {"triangle_id": i, "h1_error_sq": float(value)}
            for i, value in enumerate(materialized["smooth"]["element_h1_sq"])
        ],
    )
    write_csv(
        OUT / "singular_u128_element_h1_sq.csv",
        ["triangle_id", "h1_error_sq"],
        [
            {"triangle_id": i, "h1_error_sq": float(value)}
            for i, value in enumerate(singular_current)
        ],
    )
    write_csv(
        OUT / "materialization_check.csv",
        [
            "problem",
            "materialized_h1_error",
            "authoritative_h1_error",
            "h1_relative_difference",
            "materialized_l2_error",
            "authoritative_l2_error",
            "l2_relative_difference",
            "materialized_residual",
            "authoritative_residual",
            "residual_absolute_difference",
            "max_singular_element_absolute_difference",
            "max_singular_element_relative_difference",
        ],
        check_rows,
    )

    print("ET002-04 input materialization: PASS")
    for row in check_rows:
        print(
            f"{row['problem']}: H1_rel={float(row['h1_relative_difference']):.3e} "
            f"L2_rel={float(row['l2_relative_difference']):.3e} "
            f"residual_abs={float(row['residual_absolute_difference']):.3e}"
        )
    print(
        "singular element identity: "
        f"max_abs={float(np.max(absolute)):.3e}, "
        f"max_rel={float(np.max(relative)):.3e}"
    )


if __name__ == "__main__":
    main()
