#!/usr/bin/env python3
"""Run ET002-05: matched-DOF uniform versus frozen graded mesh.

Uniform results are read from the already-verified ET002-03 CSV.  Only the
beta=3/2 graded family is newly solved.  The experiment does not scan beta and
does not impose an outcome gate requiring graded meshes to outperform uniform
meshes.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from error_evaluator import relative_difference
from fem_core import build_l_shape_mesh, mesh_quality, solve_dirichlet_laplace
from run_et002_03 import singular_gradient, singular_value
from singular_error_evaluator import evaluate_singular_error


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/et002-05.json"
UNIFORM_CSV = ROOT / "results/et002-03/convergence.csv"
RESULT_DIR = ROOT / "results/et002-05"


def read_uniform_rows() -> dict[int, dict[str, str]]:
    with UNIFORM_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {int(row["N"]): row for row in rows}
    if len(result) != len(rows):
        raise AssertionError("ET002-03 uniform CSV contains duplicate N rows")
    return result


def observed_order(previous_error: float, current_error: float) -> float:
    return math.log(previous_error / current_error) / math.log(2.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    beta = float(config["beta"])
    if not math.isclose(beta, 1.5, rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("ET002-05 beta must remain frozen at 3/2")

    expected_N = [int(value) for value in config["uniform_N"]]
    uniform_rows = read_uniform_rows()
    if sorted(uniform_rows) != expected_N:
        raise AssertionError("ET002-03 uniform baseline inventory drifted")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    quadrature_rows: list[dict[str, object]] = []
    previous_graded_h1: float | None = None
    previous_graded_l2: float | None = None
    check_N = {int(value) for value in config["quadrature_refinement_check_N"]}

    for N in expected_N:
        uniform_mesh = build_l_shape_mesh(N, 1.0)
        graded_mesh = build_l_shape_mesh(N, beta)

        if not np.array_equal(uniform_mesh.triangles, graded_mesh.triangles):
            raise AssertionError(f"N={N}: uniform/graded connectivity mismatch")
        if not np.array_equal(uniform_mesh.boundary_nodes, graded_mesh.boundary_nodes):
            raise AssertionError(f"N={N}: boundary-node inventory mismatch")
        if uniform_mesh.points.shape[0] != graded_mesh.points.shape[0]:
            raise AssertionError(f"N={N}: node-count mismatch")
        if uniform_mesh.triangles.shape[0] != graded_mesh.triangles.shape[0]:
            raise AssertionError(f"N={N}: triangle-count mismatch")

        uniform_row = uniform_rows[N]
        if int(uniform_row["nodes"]) != uniform_mesh.points.shape[0]:
            raise AssertionError(f"N={N}: frozen uniform node count drifted")
        if int(uniform_row["triangles"]) != uniform_mesh.triangles.shape[0]:
            raise AssertionError(f"N={N}: frozen uniform triangle count drifted")

        graded_result = solve_dirichlet_laplace(graded_mesh, singular_value)
        graded_error = evaluate_singular_error(
            graded_mesh.points,
            graded_mesh.triangles,
            graded_result.values,
            singular_value,
            singular_gradient,
            noncorner_subdivision_levels=int(config["quadrature_subdivision_levels"]),
            corner_gauss_order=int(config["corner_gauss_order"]),
        )

        graded_h1_order: float | str = ""
        graded_l2_order: float | str = ""
        if previous_graded_h1 is not None and previous_graded_l2 is not None:
            graded_h1_order = observed_order(
                previous_graded_h1, graded_error.h1_seminorm
            )
            graded_l2_order = observed_order(
                previous_graded_l2, graded_error.l2
            )

        uniform_quality = mesh_quality(uniform_mesh)
        graded_quality = mesh_quality(graded_mesh)

        uniform_h1 = float(uniform_row["h1_error"])
        uniform_l2 = float(uniform_row["l2_error"])
        uniform_rel_h1 = float(uniform_row["relative_h1_error"])
        uniform_rel_l2 = float(uniform_row["relative_l2_error"])

        row = {
            "N": N,
            "beta": beta,
            "nodes": graded_mesh.points.shape[0],
            "triangles": graded_mesh.triangles.shape[0],
            "free_nodes": graded_result.free_nodes.shape[0],
            "connectivity_match": "true",
            "uniform_h1_error": uniform_h1,
            "uniform_relative_h1_error": uniform_rel_h1,
            "uniform_h1_order": uniform_row["h1_order"],
            "graded_h1_error": graded_error.h1_seminorm,
            "graded_relative_h1_error": graded_error.relative_h1_seminorm,
            "graded_h1_order": graded_h1_order,
            "graded_over_uniform_h1": graded_error.h1_seminorm / uniform_h1,
            "uniform_l2_error": uniform_l2,
            "uniform_relative_l2_error": uniform_rel_l2,
            "uniform_l2_order": uniform_row["l2_order"],
            "graded_l2_error": graded_error.l2,
            "graded_relative_l2_error": graded_error.relative_l2,
            "graded_l2_order": graded_l2_order,
            "graded_over_uniform_l2": graded_error.l2 / uniform_l2,
            "uniform_min_angle_deg": uniform_quality["min_angle_deg"],
            "graded_min_angle_deg": graded_quality["min_angle_deg"],
            "uniform_max_edge_ratio": uniform_quality["max_edge_ratio"],
            "graded_max_edge_ratio": graded_quality["max_edge_ratio"],
            "graded_reduced_residual_l2": graded_result.reduced_residual_l2,
        }
        rows.append(row)

        if graded_result.reduced_residual_l2 > float(
            config["max_reduced_residual_l2"]
        ):
            raise AssertionError(
                f"N={N}: graded residual {graded_result.reduced_residual_l2}"
            )
        if not graded_quality["min_angle_deg"] > 0.0:
            raise AssertionError(f"N={N}: non-positive graded minimum angle")
        if not math.isfinite(graded_quality["max_edge_ratio"]):
            raise AssertionError(f"N={N}: non-finite graded edge ratio")

        if N in check_N:
            refined = evaluate_singular_error(
                graded_mesh.points,
                graded_mesh.triangles,
                graded_result.values,
                singular_value,
                singular_gradient,
                noncorner_subdivision_levels=int(
                    config["quadrature_refinement_levels"]
                ),
                corner_gauss_order=int(config["corner_gauss_refinement_order"]),
            )
            h1_change = relative_difference(
                graded_error.h1_seminorm, refined.h1_seminorm
            )
            l2_change = relative_difference(graded_error.l2, refined.l2)
            quadrature_rows.append(
                {
                    "N": N,
                    "base_levels": int(config["quadrature_subdivision_levels"]),
                    "refined_levels": int(config["quadrature_refinement_levels"]),
                    "base_corner_gauss_order": int(config["corner_gauss_order"]),
                    "refined_corner_gauss_order": int(
                        config["corner_gauss_refinement_order"]
                    ),
                    "base_h1_error": graded_error.h1_seminorm,
                    "refined_h1_error": refined.h1_seminorm,
                    "relative_h1_change": h1_change,
                    "base_l2_error": graded_error.l2,
                    "refined_l2_error": refined.l2,
                    "relative_l2_change": l2_change,
                }
            )
            if max(h1_change, l2_change) > float(
                config["max_quadrature_relative_change"]
            ):
                raise AssertionError(
                    f"N={N}: graded quadrature refinement change "
                    f"{max(h1_change, l2_change):.6e} exceeds "
                    f"{float(config['max_quadrature_relative_change']):.6e}"
                )

        previous_graded_h1 = graded_error.h1_seminorm
        previous_graded_l2 = graded_error.l2

    write_csv(RESULT_DIR / "comparison.csv", rows)
    write_csv(RESULT_DIR / "quadrature_check.csv", quadrature_rows)

    fine_rows = [row for row in rows if int(row["N"]) >= 32]
    all_fine_improved = all(
        float(row["graded_h1_error"]) < float(row["uniform_h1_error"])
        for row in fine_rows
    )
    graded_orders = [
        float(row["graded_h1_order"])
        for row in rows
        if row["graded_h1_order"] != ""
    ]
    last_two_graded = graded_orders[-2:]

    print("ET002-05 PASS")
    for row in rows:
        print(
            f"N={row['N']:>3} dof={row['nodes']:>6} "
            f"uniform_H1={float(row['uniform_h1_error']):.16e} "
            f"graded_H1={float(row['graded_h1_error']):.16e} "
            f"ratio={float(row['graded_over_uniform_h1']):.8f} "
            f"graded_order={row['graded_h1_order']} "
            f"min_angle={float(row['graded_min_angle_deg']):.4f}deg "
            f"edge_ratio={float(row['graded_max_edge_ratio']):.6f}"
        )
    for row in quadrature_rows:
        print(
            f"quadrature N={row['N']}: "
            f"H1 relative change={float(row['relative_h1_change']):.3e}, "
            f"L2 relative change={float(row['relative_l2_change']):.3e}"
        )
    print(f"N>=32 all graded H1 errors lower than uniform: {all_fine_improved}")
    print(
        "last-two graded H1 orders: "
        + ", ".join(f"{value:.12f}" for value in last_two_graded)
    )


if __name__ == "__main__":
    main()
