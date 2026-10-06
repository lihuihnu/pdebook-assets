#!/usr/bin/env python3
"""Independent standard-library checker for ET002-05."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/et002-05.json"
UNIFORM = ROOT / "results/et002-03/convergence.csv"
RESULT = ROOT / "results/et002-05"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1.0e-14) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_N = [int(value) for value in config["uniform_N"]]
    beta = float(config["beta"])
    if not math.isclose(beta, 1.5, rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("ET002-05 beta drifted")

    rows = read_csv(RESULT / "comparison.csv")
    if [int(row["N"]) for row in rows] != expected_N:
        raise AssertionError("ET002-05 N inventory mismatch")

    uniform_rows = {int(row["N"]): row for row in read_csv(UNIFORM)}
    if sorted(uniform_rows) != expected_N:
        raise AssertionError("ET002-03 uniform baseline inventory mismatch")

    for row in rows:
        N = int(row["N"])
        if not math.isclose(float(row["beta"]), beta, rel_tol=0.0, abs_tol=1.0e-15):
            raise AssertionError(f"N={N}: beta mismatch")
        if row["connectivity_match"] != "true":
            raise AssertionError(f"N={N}: connectivity mismatch")
        if int(row["nodes"]) != int(uniform_rows[N]["nodes"]):
            raise AssertionError(f"N={N}: node-count mismatch")
        if int(row["triangles"]) != int(uniform_rows[N]["triangles"]):
            raise AssertionError(f"N={N}: triangle-count mismatch")

        for column, baseline_column in [
            ("uniform_h1_error", "h1_error"),
            ("uniform_relative_h1_error", "relative_h1_error"),
            ("uniform_l2_error", "l2_error"),
            ("uniform_relative_l2_error", "relative_l2_error"),
        ]:
            if not close(float(row[column]), float(uniform_rows[N][baseline_column])):
                raise AssertionError(f"N={N}: frozen uniform {column} drifted")

        if float(row["graded_reduced_residual_l2"]) > float(
            config["max_reduced_residual_l2"]
        ):
            raise AssertionError(f"N={N}: graded residual gate")
        if not float(row["graded_h1_error"]) > 0.0:
            raise AssertionError(f"N={N}: invalid graded H1 error")
        if not float(row["graded_l2_error"]) > 0.0:
            raise AssertionError(f"N={N}: invalid graded L2 error")
        if not float(row["graded_min_angle_deg"]) > 0.0:
            raise AssertionError(f"N={N}: invalid graded minimum angle")
        if not math.isfinite(float(row["graded_max_edge_ratio"])):
            raise AssertionError(f"N={N}: invalid graded edge ratio")

        expected_h1_ratio = float(row["graded_h1_error"]) / float(
            row["uniform_h1_error"]
        )
        expected_l2_ratio = float(row["graded_l2_error"]) / float(
            row["uniform_l2_error"]
        )
        if not close(float(row["graded_over_uniform_h1"]), expected_h1_ratio):
            raise AssertionError(f"N={N}: H1 ratio arithmetic mismatch")
        if not close(float(row["graded_over_uniform_l2"]), expected_l2_ratio):
            raise AssertionError(f"N={N}: L2 ratio arithmetic mismatch")

    qrows = read_csv(RESULT / "quadrature_check.csv")
    expected_q = [int(value) for value in config["quadrature_refinement_check_N"]]
    if [int(row["N"]) for row in qrows] != expected_q:
        raise AssertionError("ET002-05 quadrature inventory mismatch")
    threshold = float(config["max_quadrature_relative_change"])
    for row in qrows:
        if int(row["base_levels"]) != int(config["quadrature_subdivision_levels"]):
            raise AssertionError(f"N={row['N']}: base subdivision mismatch")
        if int(row["refined_levels"]) != int(config["quadrature_refinement_levels"]):
            raise AssertionError(f"N={row['N']}: refined subdivision mismatch")
        if int(row["base_corner_gauss_order"]) != int(config["corner_gauss_order"]):
            raise AssertionError(f"N={row['N']}: base corner Gauss mismatch")
        if int(row["refined_corner_gauss_order"]) != int(
            config["corner_gauss_refinement_order"]
        ):
            raise AssertionError(f"N={row['N']}: refined corner Gauss mismatch")
        if float(row["relative_h1_change"]) > threshold:
            raise AssertionError(f"N={row['N']}: H1 quadrature gate")
        if float(row["relative_l2_change"]) > threshold:
            raise AssertionError(f"N={row['N']}: L2 quadrature gate")

    print("ET002-05 independent CSV checker: PASS")

    fine = [row for row in rows if int(row["N"]) >= 32]
    print(
        "N>=32 all graded H1 errors lower than uniform:",
        all(
            float(row["graded_h1_error"]) < float(row["uniform_h1_error"])
            for row in fine
        ),
    )
    orders = [
        float(row["graded_h1_order"])
        for row in rows
        if row["graded_h1_order"]
    ]
    print(
        "last-two graded H1 orders:",
        ", ".join(f"{value:.12f}" for value in orders[-2:]),
    )


if __name__ == "__main__":
    main()
