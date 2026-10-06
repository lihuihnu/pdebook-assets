#!/usr/bin/env python3
"""Run ET002-04 using only frozen CSV sidecars.

This script intentionally uses the Python standard library only.  It does not
import the FEM solver or any error evaluator, and therefore cannot re-solve the
PDE.  Its sole job is spatial aggregation of frozen U128 element errors.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/et002-04.json"
INPUT_DIR = ROOT / "results/et002-04-inputs"
RESULT_DIR = ROOT / "results/et002-04"
ET02_CSV = ROOT / "results/et002-02/convergence.csv"
ET03_CSV = ROOT / "results/et002-03/convergence.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def authoritative_h1_sq(path: Path, target_n: int) -> float:
    rows = [row for row in read_csv(path) if int(row["N"]) == target_n]
    if len(rows) != 1:
        raise AssertionError(f"{path}: expected one N={target_n} row")
    value = float(rows[0]["h1_error"])
    return value * value


def element_values(path: Path, expected_count: int) -> list[float]:
    rows = read_csv(path)
    if len(rows) != expected_count:
        raise AssertionError(f"{path}: {len(rows)} rows != {expected_count}")
    ids = [int(row["triangle_id"]) for row in rows]
    if ids != list(range(expected_count)):
        raise AssertionError(f"{path}: triangle ids are not contiguous")
    values = [float(row["h1_error_sq"]) for row in rows]
    if not all(value >= 0.0 and math.isfinite(value) for value in values):
        raise AssertionError(f"{path}: invalid element error contribution")
    return values


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target_n = int(config["N"])
    radii = [float(value) for value in config["radii"]]
    threshold = float(config["max_global_sum_relative_defect"])

    if radii != sorted(radii) or len(set(radii)) != len(radii):
        raise AssertionError("ET002-04 radii must be strictly increasing and unique")
    if any(radius <= 0.0 for radius in radii):
        raise AssertionError("ET002-04 radii must be positive")

    triangles = read_csv(INPUT_DIR / "u128_triangles.csv")
    if len(triangles) != 24576:
        raise AssertionError(f"triangle inventory {len(triangles)} != 24576")

    ids = [int(row["triangle_id"]) for row in triangles]
    if ids != list(range(len(triangles))):
        raise AssertionError("triangle geometry ids are not contiguous")

    smooth = element_values(INPUT_DIR / "smooth_u128_element_h1_sq.csv", len(triangles))
    singular = element_values(INPUT_DIR / "singular_u128_element_h1_sq.csv", len(triangles))
    centroid_r = [float(row["centroid_r"]) for row in triangles]

    totals = {
        "smooth": math.fsum(smooth),
        "singular": math.fsum(singular),
    }
    authoritative = {
        "smooth": authoritative_h1_sq(ET02_CSV, target_n),
        "singular": authoritative_h1_sq(ET03_CSV, target_n),
    }

    summary_rows: list[dict[str, object]] = []
    for problem in ("smooth", "singular"):
        defect = abs(totals[problem] - authoritative[problem]) / max(
            authoritative[problem], 1.0e-300
        )
        if defect > threshold:
            raise AssertionError(
                f"{problem}: element/global H1^2 defect {defect} > {threshold}"
            )
        summary_rows.append(
            {
                "problem": problem,
                "N": target_n,
                "total_element_h1_sq": totals[problem],
                "authoritative_h1_sq": authoritative[problem],
                "relative_sum_defect": defect,
                "h1_error": math.sqrt(totals[problem]),
            }
        )

    cumulative_rows: list[dict[str, object]] = []
    previous = {"smooth": -1.0, "singular": -1.0}
    for radius in radii:
        inside_indices = [
            index for index, value in enumerate(centroid_r) if value < radius
        ]
        count = len(inside_indices)
        row: dict[str, object] = {
            "R": radius,
            "inside_triangles": count,
            "total_triangles": len(triangles),
        }
        for problem, values in (("smooth", smooth), ("singular", singular)):
            inside_sq = math.fsum(values[index] for index in inside_indices)
            fraction = inside_sq / totals[problem]
            if fraction + 1.0e-15 < previous[problem]:
                raise AssertionError(f"{problem}: cumulative fraction is not monotone")
            if not (0.0 <= fraction <= 1.0 + 1.0e-15):
                raise AssertionError(f"{problem}: invalid cumulative fraction {fraction}")
            previous[problem] = fraction
            row[f"{problem}_inside_h1_sq"] = inside_sq
            row[f"{problem}_C"] = fraction
        row["singular_minus_smooth_C"] = (
            float(row["singular_C"]) - float(row["smooth_C"])
        )
        cumulative_rows.append(row)

    spatial_rows: list[dict[str, object]] = []
    for index, tri in enumerate(triangles):
        spatial_rows.append(
            {
                "triangle_id": index,
                "centroid_x": float(tri["centroid_x"]),
                "centroid_y": float(tri["centroid_y"]),
                "centroid_r": centroid_r[index],
                "area": float(tri["area"]),
                "smooth_h1_error_sq": smooth[index],
                "singular_h1_error_sq": singular[index],
                "smooth_fraction": smooth[index] / totals["smooth"],
                "singular_fraction": singular[index] / totals["singular"],
            }
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULT_DIR / "summary.csv",
        [
            "problem",
            "N",
            "total_element_h1_sq",
            "authoritative_h1_sq",
            "relative_sum_defect",
            "h1_error",
        ],
        summary_rows,
    )
    write_csv(
        RESULT_DIR / "cumulative.csv",
        [
            "R",
            "inside_triangles",
            "total_triangles",
            "smooth_inside_h1_sq",
            "smooth_C",
            "singular_inside_h1_sq",
            "singular_C",
            "singular_minus_smooth_C",
        ],
        cumulative_rows,
    )
    write_csv(
        RESULT_DIR / "element_spatial.csv",
        [
            "triangle_id",
            "centroid_x",
            "centroid_y",
            "centroid_r",
            "area",
            "smooth_h1_error_sq",
            "singular_h1_error_sq",
            "smooth_fraction",
            "singular_fraction",
        ],
        spatial_rows,
    )

    print("ET002-04 PASS")
    for row in cumulative_rows:
        print(
            f"R={float(row['R']):.3f}: "
            f"smooth_C={float(row['smooth_C']):.12f} "
            f"singular_C={float(row['singular_C']):.12f} "
            f"delta={float(row['singular_minus_smooth_C']):.12f}"
        )


if __name__ == "__main__":
    main()
