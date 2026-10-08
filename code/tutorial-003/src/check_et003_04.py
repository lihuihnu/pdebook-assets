#!/usr/bin/env python3
"""Independent standard-library checker for ET003-04."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


# Locate frozen configs/results relative to this source file. This works in
# private experiments/tutorial-003/ and public code/tutorial-003/ alike.
BASE = Path(__file__).resolve().parent.parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config() -> dict:
    return json.loads(
        (BASE / "config/et003-04.json").read_text(encoding="utf-8")
    )


def total_variation(values: list[float]) -> float:
    return sum(abs(b - a) for a, b in zip(values, values[1:]))


def monotonicity_violation_edges(
    values: list[float],
    *,
    tolerance: float,
) -> list[tuple[str, int, int, float]]:
    N = len(values) - 1
    result: list[tuple[str, int, int, float]] = []
    for edge in range(N):
        delta = values[edge + 1] - values[edge]
        midpoint = (edge + 0.5) / N
        if midpoint < 0.5 and delta < -tolerance:
            result.append(("left", edge, edge + 1, delta))
        elif midpoint > 0.5 and delta > tolerance:
            result.append(("right", edge, edge + 1, delta))
    return result


def discrete_l2(error: list[float], h: float) -> float:
    return math.sqrt(h * sum(value * value for value in error))


def cn_amplification(xi: float) -> float:
    return (1.0 - 0.5 * xi) / (1.0 + 0.5 * xi)


def main() -> None:
    config = load_config()
    N = int(config["N"])
    h = 1.0 / N
    r_values = [float(value) for value in config["r_values"]]
    tolerance = float(config["monotonicity_tolerance"])

    scan = read_csv(BASE / "results/et003-04/scan.csv")
    profiles = read_csv(BASE / "results/et003-04/profiles.csv")
    violation_rows = read_csv(
        BASE / "results/et003-04/monotonicity_violations.csv"
    )

    methods = [
        "semidiscrete_reference",
        "backward_euler",
        "crank_nicolson",
        "rannacher_r1",
    ]

    if len(scan) != len(r_values) * len(methods):
        raise AssertionError("ET003-04 scan inventory mismatch")
    if len(profiles) != len(r_values) * (N + 1):
        raise AssertionError("ET003-04 profile inventory mismatch")

    r_flip = 1.0 / (
        2.0 * math.cos(math.pi / (2.0 * N)) ** 2
    )
    xi_factor = 4.0 * math.cos(math.pi / (2.0 * N)) ** 2

    scan_map = {
        (row["method"], float(row["r"])): row for row in scan
    }
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in profiles:
        grouped.setdefault(float(row["r"]), []).append(row)

    expected_violations: set[tuple[float, str, str, int, int]] = set()

    for r in r_values:
        rows = grouped.get(r)
        if rows is None or len(rows) != N + 1:
            raise AssertionError(f"r={r}: profile inventory")
        rows.sort(key=lambda row: int(row["node_index"]))

        dt = r * h * h
        xi_max = xi_factor * r
        g_max = cn_amplification(xi_max)

        for index, row in enumerate(rows):
            if int(row["node_index"]) != index:
                raise AssertionError(f"r={r}: node index")
            if abs(float(row["x"]) - index * h) > 2.0e-15:
                raise AssertionError(f"r={r}: x grid")
            if abs(float(row["dt"]) - dt) > 2.0e-18:
                raise AssertionError(f"r={r}: dt")

        reference = [
            float(row["semidiscrete_reference"]) for row in rows
        ]

        for method in methods:
            values = [float(row[method]) for row in rows]
            error = [value - ref for value, ref in zip(values, reference)]
            violations = monotonicity_violation_edges(
                values,
                tolerance=tolerance,
            )
            stored = scan_map[(method, r)]

            if abs(float(stored["r_flip"]) - r_flip) > 2.0e-15:
                raise AssertionError(f"r={r}: r_flip")
            if abs(float(stored["xi_max"]) - xi_max) > 2.0e-13:
                raise AssertionError(f"r={r}: xi_max")
            if abs(
                float(stored["cn_highest_mode_amplification"]) - g_max
            ) > 2.0e-14:
                raise AssertionError(f"r={r}: CN highest-mode G")
            if int(stored["highest_mode_sign_flipped"]) != int(g_max < 0.0):
                raise AssertionError(f"r={r}: flip flag")

            comparisons = {
                "l2_error_vs_semidiscrete": discrete_l2(error, h),
                "linf_error_vs_semidiscrete": max(abs(value) for value in error),
                "total_variation": total_variation(values),
                "min_value": min(values),
                "max_value": max(values),
            }
            for key, expected in comparisons.items():
                observed = float(stored[key])
                scale = max(abs(expected), 1.0)
                if abs(observed - expected) > 2.0e-13 * scale:
                    raise AssertionError(f"r={r} {method}: {key}")

            if int(stored["monotonicity_violation_count"]) != len(violations):
                raise AssertionError(f"r={r} {method}: violation count")

            for side, left, right, _ in violations:
                expected_violations.add((r, method, side, left, right))

    stored_violations = {
        (
            float(row["r"]),
            row["method"],
            row["side"],
            int(row["edge_left_index"]),
            int(row["edge_right_index"]),
        )
        for row in violation_rows
    }
    if stored_violations != expected_violations:
        raise AssertionError("ET003-04 violation location inventory mismatch")

    for r in r_values:
        for method, key in (
            ("semidiscrete_reference", "max_reference_violation_count"),
            ("backward_euler", "max_be_violation_count"),
            ("rannacher_r1", "max_r1_violation_count"),
        ):
            if int(scan_map[(method, r)][
                "monotonicity_violation_count"
            ]) > int(config[key]):
                raise AssertionError(f"r={r} {method}: monotonicity gate")

    pressure_r = float(config["pressure_r"])
    if int(scan_map[("crank_nicolson", pressure_r)][
        "monotonicity_violation_count"
    ]) < int(config["min_cn_pressure_violation_count"]):
        raise AssertionError("pressure-r CN response missing")

    # The theoretical finite-N threshold must lie between the frozen 0.5 and 1
    # samples.  This checks the scientific distinction without declaring the
    # first visible zigzag in advance.
    if not 0.5 < r_flip < 1.0:
        raise AssertionError(f"unexpected finite-N r_flip={r_flip}")

    print("ET003-04 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
