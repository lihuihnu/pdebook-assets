#!/usr/bin/env python3
"""Run ET002-03: re-entrant-corner singular harmonic benchmark.

The already-verified FEM core and independent error evaluator are reused
unchanged.  This driver adds only the frozen singular exact solution and the
ET002-03 acceptance logic.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from error_evaluator import relative_difference
from fem_core import build_l_shape_mesh, solve_dirichlet_laplace
from singular_error_evaluator import evaluate_singular_error


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/et002-03.json"
SMOOTH_CSV = ROOT / "results/et002-02/convergence.csv"
RESULT_DIR = ROOT / "results/et002-03"


def _polar_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return theta in [0, 2*pi), matching the frozen L-domain convention."""
    theta = np.arctan2(y, x)
    return np.where(theta < 0.0, theta + 2.0 * np.pi, theta)


def singular_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lam = 2.0 / 3.0
    r = np.hypot(x, y)
    theta = _polar_angle(x, y)
    value = np.zeros_like(r, dtype=float)
    mask = r > 0.0
    value[mask] = np.power(r[mask], lam) * np.sin(lam * theta[mask])
    return value


def singular_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact Cartesian gradient away from the re-entrant corner.

    Quadrature points are strictly interior to virtual subtriangles, so the
    evaluator never requests the gradient at r=0.
    """
    lam = 2.0 / 3.0
    r = np.hypot(x, y)
    if np.any(r <= 0.0):
        raise ValueError("singular gradient is undefined at r=0")

    theta = _polar_angle(x, y)
    factor = lam * np.power(r, lam - 1.0)
    phase = (lam - 1.0) * theta
    gx = factor * np.sin(phase)
    gy = factor * np.cos(phase)
    return np.column_stack((gx, gy))


def observed_order(previous_error: float, current_error: float) -> float:
    return math.log(previous_error / current_error) / math.log(2.0)


def _read_smooth_last_two_orders() -> list[float]:
    with SMOOTH_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    orders = [float(row["h1_order"]) for row in rows if row["h1_order"]]
    if len(orders) < 2:
        raise AssertionError("ET002-02 baseline does not contain two H1 orders")
    return orders[-2:]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lam = float(config["lambda"])
    if not math.isclose(lam, 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("ET002-03 lambda must remain frozen at 2/3")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    element_rows: list[dict[str, object]] = []
    gate_failures: list[str] = []

    previous_h1: float | None = None
    previous_l2: float | None = None
    check_N = {int(value) for value in config["quadrature_refinement_check_N"]}

    for N_value in config["uniform_N"]:
        N = int(N_value)
        mesh = build_l_shape_mesh(N, 1.0)
        result = solve_dirichlet_laplace(mesh, singular_value)

        error = evaluate_singular_error(
            mesh.points,
            mesh.triangles,
            result.values,
            singular_value,
            singular_gradient,
            noncorner_subdivision_levels=int(config["quadrature_subdivision_levels"]),
            corner_gauss_order=int(config["corner_gauss_order"]),
            return_element_h1_sq=True,
        )
        if error.element_h1_sq is None:
            raise AssertionError("ET002-03 requires element H1 contributions")

        element_sum = float(np.sum(error.element_h1_sq))
        global_sq = float(error.h1_seminorm * error.h1_seminorm)
        element_sum_defect = abs(element_sum - global_sq) / max(global_sq, 1.0e-300)

        h1_order: float | str = ""
        l2_order: float | str = ""
        if previous_h1 is not None and previous_l2 is not None:
            h1_order = observed_order(previous_h1, error.h1_seminorm)
            l2_order = observed_order(previous_l2, error.l2)

        rows.append(
            {
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
                "element_sum_relative_defect": element_sum_defect,
            }
        )

        for triangle_id, contribution in enumerate(error.element_h1_sq):
            element_rows.append(
                {
                    "N": N,
                    "triangle_id": triangle_id,
                    "h1_error_sq": float(contribution),
                }
            )

        if result.reduced_residual_l2 > float(config["max_reduced_residual_l2"]):
            gate_failures.append(
                f"N={N}: residual {result.reduced_residual_l2:.6e} exceeds "
                f"{float(config['max_reduced_residual_l2']):.6e}"
            )

        if element_sum_defect > float(config["max_element_sum_relative_defect"]):
            gate_failures.append(
                f"N={N}: element H1 sum defect {element_sum_defect:.6e} exceeds "
                f"{float(config['max_element_sum_relative_defect']):.6e}"
            )

        if N in check_N:
            refined = evaluate_singular_error(
                mesh.points,
                mesh.triangles,
                result.values,
                singular_value,
                singular_gradient,
                noncorner_subdivision_levels=int(config["quadrature_refinement_levels"]),
                corner_gauss_order=int(config["corner_gauss_refinement_order"]),
            )
            h1_change = relative_difference(error.h1_seminorm, refined.h1_seminorm)
            l2_change = relative_difference(error.l2, refined.l2)
            refinement_rows.append(
                {
                    "N": N,
                    "base_levels": int(config["quadrature_subdivision_levels"]),
                    "refined_levels": int(config["quadrature_refinement_levels"]),
                    "base_corner_gauss_order": int(config["corner_gauss_order"]),
                    "refined_corner_gauss_order": int(config["corner_gauss_refinement_order"]),
                    "base_h1_error": error.h1_seminorm,
                    "refined_h1_error": refined.h1_seminorm,
                    "relative_h1_change": h1_change,
                    "base_l2_error": error.l2,
                    "refined_l2_error": refined.l2,
                    "relative_l2_change": l2_change,
                }
            )
            if max(h1_change, l2_change) > float(config["max_quadrature_relative_change"]):
                gate_failures.append(
                    f"N={N}: quadrature refinement changes error norm by "
                    f"{max(h1_change, l2_change):.6e}, exceeding "
                    f"{float(config['max_quadrature_relative_change']):.6e}"
                )

        previous_h1 = error.h1_seminorm
        previous_l2 = error.l2

    _write_csv(RESULT_DIR / "convergence.csv", rows)
    _write_csv(RESULT_DIR / "quadrature_check.csv", refinement_rows)
    _write_csv(RESULT_DIR / "element_h1_sq.csv", element_rows)

    h1_values = [float(row["h1_error"]) for row in rows]
    if not all(a > b > 0.0 for a, b in zip(h1_values, h1_values[1:])):
        gate_failures.append("H1 error is not strictly decreasing")

    numeric_orders = [float(row["h1_order"]) for row in rows if row["h1_order"] != ""]
    low, high = (float(value) for value in config["h1_order_window"])
    last_two = numeric_orders[-2:]
    if len(last_two) != 2 or not all(low <= value <= high for value in last_two):
        gate_failures.append(
            f"last two H1 orders {last_two} outside frozen window [{low}, {high}]"
        )

    smooth_last_two = _read_smooth_last_two_orders()
    singular_mean = sum(last_two) / 2.0
    smooth_mean = sum(smooth_last_two) / 2.0
    order_drop = smooth_mean - singular_mean
    if order_drop < float(config["min_mean_order_drop_vs_et002_02"]):
        gate_failures.append(
            f"mean H1-order drop vs ET002-02 is {order_drop:.6f}, below frozen "
            f"minimum {float(config['min_mean_order_drop_vs_et002_02']):.6f}"
        )

    print("ET002-03 RESULTS")
    for row in rows:
        print(
            f"N={row['N']:>3} nodes={row['nodes']:>6} "
            f"H1={row['h1_error']:.16e} order={row['h1_order']} "
            f"L2={row['l2_error']:.16e} order={row['l2_order']} "
            f"residual={row['reduced_residual_l2']:.16e} "
            f"element_sum_defect={row['element_sum_relative_defect']:.3e}"
        )
    for row in refinement_rows:
        print(
            f"quadrature N={row['N']}: "
            f"H1 relative change={row['relative_h1_change']:.6e}, "
            f"L2 relative change={row['relative_l2_change']:.6e}"
        )
    print(
        "last-two mean H1 order: "
        f"singular={singular_mean:.12f}, smooth={smooth_mean:.12f}, "
        f"drop={order_drop:.12f}"
    )

    if gate_failures:
        print("ET002-03 GATE FAILURES")
        for failure in gate_failures:
            print(f"- {failure}")
        raise AssertionError("ET002-03 failed one or more frozen gates")

    print("ET002-03 PASS")


if __name__ == "__main__":
    main()
