#!/usr/bin/env python3
"""Standard-library checker for materialized ET002-04 frozen inputs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
OUT = RESULT_ROOT / "et002-04-inputs"

METRIC_REL_TOL = 1.0e-12
RESIDUAL_ABS_TOL = 1.0e-13
ELEMENT_REL_TOL = 1.0e-12
ELEMENT_ABS_TOL = 1.0e-18


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    nodes = read_csv(OUT / "u128_nodes.csv")
    triangles = read_csv(OUT / "u128_triangles.csv")
    smooth_solution = read_csv(OUT / "smooth_u128_solution.csv")
    singular_solution = read_csv(OUT / "singular_u128_solution.csv")
    smooth_elements = read_csv(OUT / "smooth_u128_element_h1_sq.csv")
    singular_elements = read_csv(OUT / "singular_u128_element_h1_sq.csv")
    checks = read_csv(OUT / "materialization_check.csv")

    if len(nodes) != 12545:
        raise AssertionError(f"node count {len(nodes)} != 12545")
    if len(triangles) != 24576:
        raise AssertionError(f"triangle count {len(triangles)} != 24576")
    if len(smooth_solution) != len(nodes) or len(singular_solution) != len(nodes):
        raise AssertionError("solution/node inventory mismatch")
    if len(smooth_elements) != len(triangles) or len(singular_elements) != len(triangles):
        raise AssertionError("element/triangle inventory mismatch")

    expected_node_ids = [str(i) for i in range(len(nodes))]
    if [row["node_id"] for row in nodes] != expected_node_ids:
        raise AssertionError("node ids are not contiguous")
    if [row["node_id"] for row in smooth_solution] != expected_node_ids:
        raise AssertionError("smooth solution node ids drifted")
    if [row["node_id"] for row in singular_solution] != expected_node_ids:
        raise AssertionError("singular solution node ids drifted")

    expected_tri_ids = [str(i) for i in range(len(triangles))]
    if [row["triangle_id"] for row in triangles] != expected_tri_ids:
        raise AssertionError("triangle ids are not contiguous")
    if [row["triangle_id"] for row in smooth_elements] != expected_tri_ids:
        raise AssertionError("smooth element ids drifted")
    if [row["triangle_id"] for row in singular_elements] != expected_tri_ids:
        raise AssertionError("singular element ids drifted")

    if {row["problem"] for row in checks} != {"smooth", "singular"}:
        raise AssertionError("materialization check inventory mismatch")
    for row in checks:
        if float(row["h1_relative_difference"]) > METRIC_REL_TOL:
            raise AssertionError(f"{row['problem']}: H1 identity gate")
        if float(row["l2_relative_difference"]) > METRIC_REL_TOL:
            raise AssertionError(f"{row['problem']}: L2 identity gate")
        if float(row["residual_absolute_difference"]) > RESIDUAL_ABS_TOL:
            raise AssertionError(f"{row['problem']}: residual identity gate")

    singular_check = next(row for row in checks if row["problem"] == "singular")
    if (
        float(singular_check["max_singular_element_absolute_difference"]) > ELEMENT_ABS_TOL
        and float(singular_check["max_singular_element_relative_difference"]) > ELEMENT_REL_TOL
    ):
        raise AssertionError("singular element identity gate")

    print("ET002-04 frozen-input checker: PASS")


if __name__ == "__main__":
    main()
