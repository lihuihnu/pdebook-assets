#!/usr/bin/env python3
"""Independent CSV-level acceptance checker for ET002-01 and ET002-02.

This checker uses only the Python standard library and does not import the FEM
solver, SciPy, NumPy or the error evaluator.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_et002_01() -> None:
    config = json.loads(
        (ROOT / "config/et002-01.json").read_text(encoding="utf-8")
    )
    rows = read_csv(ROOT / "results/et002-01/summary.csv")
    expected = [entry["name"] for entry in config["meshes"]]
    if [row["mesh"] for row in rows] != expected:
        raise AssertionError("ET002-01 mesh inventory mismatch")

    for row in rows:
        if float(row["min_signed_area"]) <= 0.0:
            raise AssertionError(f"{row['mesh']}: non-positive area")
        if float(row["max_nodal_error"]) > float(config["max_nodal_error"]):
            raise AssertionError(f"{row['mesh']}: patch error")
        if float(row["reduced_residual_l2"]) > float(config["max_reduced_residual_l2"]):
            raise AssertionError(f"{row['mesh']}: residual")
        if float(row["max_local_symmetry_defect"]) > float(config["max_local_symmetry_defect"]):
            raise AssertionError(f"{row['mesh']}: local symmetry")
        if float(row["max_local_row_sum_defect"]) > float(config["max_local_row_sum_defect"]):
            raise AssertionError(f"{row['mesh']}: row sum")
        if float(row["free_block_symmetry_defect"]) > float(config["max_free_block_symmetry_defect"]):
            raise AssertionError(f"{row['mesh']}: free symmetry")


def check_et002_02() -> None:
    config = json.loads(
        (ROOT / "config/et002-02.json").read_text(encoding="utf-8")
    )
    rows = read_csv(ROOT / "results/et002-02/convergence.csv")
    expected_N = [int(value) for value in config["uniform_N"]]
    if [int(row["N"]) for row in rows] != expected_N:
        raise AssertionError("ET002-02 N inventory mismatch")

    h1 = [float(row["h1_error"]) for row in rows]
    l2 = [float(row["l2_error"]) for row in rows]
    if not all(a > b > 0.0 for a, b in zip(h1, h1[1:])):
        raise AssertionError("ET002-02 H1 error is not strictly decreasing")
    if not all(a > b > 0.0 for a, b in zip(l2, l2[1:])):
        raise AssertionError("ET002-02 L2 error is not strictly decreasing")

    orders = []
    for previous, current in zip(h1, h1[1:]):
        orders.append(math.log(previous / current) / math.log(2.0))

    low, high = (float(value) for value in config["h1_order_window"])
    if not all(low <= value <= high for value in orders[-2:]):
        raise AssertionError(f"ET002-02 last two H1 orders outside [{low}, {high}]")

    for row in rows:
        if float(row["reduced_residual_l2"]) > float(config["max_reduced_residual_l2"]):
            raise AssertionError(f"N={row['N']}: residual")

    qrows = read_csv(ROOT / "results/et002-02/quadrature_check.csv")
    expected_q = [int(value) for value in config["quadrature_refinement_check_N"]]
    if [int(row["N"]) for row in qrows] != expected_q:
        raise AssertionError("quadrature-check inventory mismatch")

    threshold = float(config["max_quadrature_relative_change"])
    for row in qrows:
        if float(row["relative_h1_change"]) > threshold:
            raise AssertionError(f"N={row['N']}: H1 quadrature refinement")
        if float(row["relative_l2_change"]) > threshold:
            raise AssertionError(f"N={row['N']}: L2 quadrature refinement")


def main() -> None:
    check_et002_01()
    check_et002_02()
    print("ET002-01/02 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
