#!/usr/bin/env python3
"""Run ET002-02: smooth harmonic control on the uniform mesh family."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from error_evaluator import evaluate_error, relative_difference
from fem_core import build_l_shape_mesh, solve_dirichlet_laplace


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/et002-02.json"
RESULT_DIR = ROOT / "results/et002-02"


def smooth_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * x - y * y


def smooth_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack((2.0 * x, -2.0 * y))


def observed_order(previous_error: float, current_error: float) -> float:
    return math.log(previous_error / current_error) / math.log(2.0)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    previous_h1: float | None = None
    previous_l2: float | None = None

    refinement_rows: list[dict[str, object]] = []

    for N in config["uniform_N"]:
        N = int(N)
        mesh = build_l_shape_mesh(N, 1.0)
        result = solve_dirichlet_laplace(mesh, smooth_value)

        error = evaluate_error(
            mesh.points,
            mesh.triangles,
            result.values,
            smooth_value,
            smooth_gradient,
            subdivision_levels=int(config["quadrature_subdivision_levels"]),
        )

        h1_order = ""
        l2_order = ""
        if previous_h1 is not None:
            h1_order = observed_order(previous_h1, error.h1_seminorm)
            l2_order = observed_order(previous_l2, error.l2)

        row = {
            "N": N,
            "h": 2.0 / N,
            "nodes": mesh.points.shape[0],
            "triangles": mesh.triangles.shape[0],
            "free_nodes": result.free_nodes.shape[0],
            "h1_error": error.h1_seminorm,
            "relative_h1_error": error.relative_h1_seminorm,
            "h1_order": h1_order,
            "l2_error": error.l2,
            "relative_l2_error": error.relative_l2,
            "l2_order": l2_order,
            "reduced_residual_l2": result.reduced_residual_l2,
        }
        rows.append(row)

        if result.reduced_residual_l2 > float(config["max_reduced_residual_l2"]):
            raise AssertionError(f"N={N}: residual {result.reduced_residual_l2}")

        if N in {int(v) for v in config["quadrature_refinement_check_N"]}:
            refined = evaluate_error(
                mesh.points,
                mesh.triangles,
                result.values,
                smooth_value,
                smooth_gradient,
                subdivision_levels=int(config["quadrature_refinement_levels"]),
            )
            h1_change = relative_difference(error.h1_seminorm, refined.h1_seminorm)
            l2_change = relative_difference(error.l2, refined.l2)
            refinement_rows.append(
                {
                    "N": N,
                    "base_levels": int(config["quadrature_subdivision_levels"]),
                    "refined_levels": int(config["quadrature_refinement_levels"]),
                    "base_h1_error": error.h1_seminorm,
                    "refined_h1_error": refined.h1_seminorm,
                    "relative_h1_change": h1_change,
                    "base_l2_error": error.l2,
                    "refined_l2_error": refined.l2,
                    "relative_l2_change": l2_change,
                }
            )
            if max(h1_change, l2_change) > float(config["max_quadrature_relative_change"]):
                raise AssertionError(
                    f"N={N}: quadrature refinement changed an error norm too much"
                )

        previous_h1 = error.h1_seminorm
        previous_l2 = error.l2

    numeric_orders = [float(row["h1_order"]) for row in rows if row["h1_order"] != ""]
    low, high = (float(v) for v in config["h1_order_window"])
    last_two = numeric_orders[-2:]
    if len(last_two) != 2 or not all(low <= value <= high for value in last_two):
        raise AssertionError(
            f"last two H1 orders {last_two} outside frozen window [{low}, {high}]"
        )

    fieldnames = list(rows[0].keys())
    with (RESULT_DIR / "convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if refinement_rows:
        fields = list(refinement_rows[0].keys())
        with (RESULT_DIR / "quadrature_check.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(refinement_rows)

    print("ET002-02 PASS")
    for row in rows:
        print(
            f"N={row['N']:>3} nodes={row['nodes']:>6} "
            f"H1={row['h1_error']:.16e} order={row['h1_order']} "
            f"L2={row['l2_error']:.16e} order={row['l2_order']} "
            f"residual={row['reduced_residual_l2']:.16e}"
        )
    for row in refinement_rows:
        print(
            f"quadrature N={row['N']}: "
            f"H1 relative change={row['relative_h1_change']:.3e}, "
            f"L2 relative change={row['relative_l2_change']:.3e}"
        )


if __name__ == "__main__":
    main()
