#!/usr/bin/env python3
"""Independent CSV-only acceptance checker for ET002-03.

Uses only the Python standard library.  It reads committed/generated result
CSVs plus the frozen ET002-02 smooth baseline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/et002-03.json"
RESULT_DIR = ROOT / "results/et002-03"
SMOOTH_CSV = ROOT / "results/et002-02/convergence.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def observed_orders(errors: list[float]) -> list[float]:
    return [
        math.log(previous / current) / math.log(2.0)
        for previous, current in zip(errors, errors[1:])
    ]


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_csv(RESULT_DIR / "convergence.csv")
    expected_N = [int(value) for value in config["uniform_N"]]
    actual_N = [int(row["N"]) for row in rows]
    if actual_N != expected_N:
        raise AssertionError(f"ET002-03 N inventory mismatch: {actual_N}")

    h1 = [float(row["h1_error"]) for row in rows]
    l2 = [float(row["l2_error"]) for row in rows]
    if not all(a > b > 0.0 for a, b in zip(h1, h1[1:])):
        raise AssertionError("ET002-03 H1 error is not strictly decreasing")
    if not all(a > b > 0.0 for a, b in zip(l2, l2[1:])):
        raise AssertionError("ET002-03 L2 error is not strictly decreasing")

    h1_orders = observed_orders(h1)
    low, high = (float(value) for value in config["h1_order_window"])
    if not all(low <= value <= high for value in h1_orders[-2:]):
        raise AssertionError(
            f"ET002-03 last two H1 orders {h1_orders[-2:]} outside [{low}, {high}]"
        )

    for row in rows:
        if float(row["reduced_residual_l2"]) > float(config["max_reduced_residual_l2"]):
            raise AssertionError(f"N={row['N']}: residual gate failed")
        if float(row["element_sum_relative_defect"]) > float(
            config["max_element_sum_relative_defect"]
        ):
            raise AssertionError(f"N={row['N']}: element-sum gate failed")

    smooth_rows = read_csv(SMOOTH_CSV)
    smooth_h1 = [float(row["h1_error"]) for row in smooth_rows]
    smooth_orders = observed_orders(smooth_h1)
    singular_mean = sum(h1_orders[-2:]) / 2.0
    smooth_mean = sum(smooth_orders[-2:]) / 2.0
    drop = smooth_mean - singular_mean
    if drop < float(config["min_mean_order_drop_vs_et002_02"]):
        raise AssertionError(
            f"H1-order drop {drop} below frozen minimum "
            f"{config['min_mean_order_drop_vs_et002_02']}"
        )

    qrows = read_csv(RESULT_DIR / "quadrature_check.csv")
    expected_q = [int(value) for value in config["quadrature_refinement_check_N"]]
    if [int(row["N"]) for row in qrows] != expected_q:
        raise AssertionError("ET002-03 quadrature inventory mismatch")
    q_threshold = float(config["max_quadrature_relative_change"])
    for row in qrows:
        if int(row["base_levels"]) != int(config["quadrature_subdivision_levels"]):
            raise AssertionError(f"N={row['N']}: base subdivision level mismatch")
        if int(row["refined_levels"]) != int(config["quadrature_refinement_levels"]):
            raise AssertionError(f"N={row['N']}: refined subdivision level mismatch")
        if int(row["base_corner_gauss_order"]) != int(config["corner_gauss_order"]):
            raise AssertionError(f"N={row['N']}: base corner Gauss order mismatch")
        if int(row["refined_corner_gauss_order"]) != int(
            config["corner_gauss_refinement_order"]
        ):
            raise AssertionError(f"N={row['N']}: refined corner Gauss order mismatch")
        if float(row["relative_h1_change"]) > q_threshold:
            raise AssertionError(f"N={row['N']}: H1 quadrature gate failed")
        if float(row["relative_l2_change"]) > q_threshold:
            raise AssertionError(f"N={row['N']}: L2 quadrature gate failed")

    erows = read_csv(RESULT_DIR / "element_h1_sq.csv")
    expected_count = sum(int(row["triangles"]) for row in rows)
    if len(erows) != expected_count:
        raise AssertionError(
            f"element contribution count {len(erows)} != {expected_count}"
        )

    by_N: dict[int, float] = {N: 0.0 for N in expected_N}
    for row in erows:
        by_N[int(row["N"])] += float(row["h1_error_sq"])
    for row in rows:
        N = int(row["N"])
        target = float(row["h1_error"]) ** 2
        defect = abs(by_N[N] - target) / max(target, 1.0e-300)
        if defect > float(config["max_element_sum_relative_defect"]):
            raise AssertionError(f"N={N}: independent element-sum defect {defect}")

    print("ET002-03 independent CSV checker: PASS")
    print(
        "last-two H1 orders:",
        ", ".join(f"{value:.12f}" for value in h1_orders[-2:]),
    )
    print(
        f"mean order drop vs ET002-02: {drop:.12f}"
    )


if __name__ == "__main__":
    main()
