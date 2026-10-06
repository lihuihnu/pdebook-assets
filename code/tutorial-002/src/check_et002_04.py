#!/usr/bin/env python3
"""Independent standard-library checker for ET002-04 spatial statistics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/et002-04.json"
RESULT = ROOT / "results/et002-04"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    radii = [float(value) for value in config["radii"]]
    threshold = float(config["max_global_sum_relative_defect"])

    summary = read_csv(RESULT / "summary.csv")
    if [row["problem"] for row in summary] != ["smooth", "singular"]:
        raise AssertionError("ET002-04 summary inventory mismatch")
    for row in summary:
        if float(row["relative_sum_defect"]) > threshold:
            raise AssertionError(f"{row['problem']}: global sum defect gate")
        if not float(row["h1_error"]) > 0.0:
            raise AssertionError(f"{row['problem']}: non-positive H1 error")

    cumulative = read_csv(RESULT / "cumulative.csv")
    if [float(row["R"]) for row in cumulative] != radii:
        raise AssertionError("ET002-04 radius inventory mismatch")

    previous = {"smooth": -1.0, "singular": -1.0}
    previous_count = -1
    for row in cumulative:
        count = int(row["inside_triangles"])
        if count < previous_count:
            raise AssertionError("inside triangle count is not monotone")
        previous_count = count
        if int(row["total_triangles"]) != 24576:
            raise AssertionError("total triangle inventory mismatch")
        for problem in ("smooth", "singular"):
            fraction = float(row[f"{problem}_C"])
            if not (0.0 <= fraction <= 1.0 + 1.0e-15):
                raise AssertionError(f"{problem}: invalid cumulative fraction")
            if fraction + 1.0e-15 < previous[problem]:
                raise AssertionError(f"{problem}: cumulative fraction not monotone")
            previous[problem] = fraction
        expected_delta = float(row["singular_C"]) - float(row["smooth_C"])
        if not math.isclose(
            float(row["singular_minus_smooth_C"]),
            expected_delta,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise AssertionError("cumulative contrast arithmetic mismatch")

    spatial = read_csv(RESULT / "element_spatial.csv")
    if len(spatial) != 24576:
        raise AssertionError("ET002-04 element spatial inventory mismatch")
    if [int(row["triangle_id"]) for row in spatial] != list(range(24576)):
        raise AssertionError("ET002-04 spatial triangle ids are not contiguous")

    for problem in ("smooth", "singular"):
        fraction_sum = math.fsum(float(row[f"{problem}_fraction"]) for row in spatial)
        if abs(fraction_sum - 1.0) > 1.0e-12:
            raise AssertionError(f"{problem}: element fractions do not sum to one")

    print("ET002-04 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
