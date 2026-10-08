#!/usr/bin/env python3
"""Independent standard-library checker for ET003-03."""

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
        (BASE / "config/et003-03.json").read_text(encoding="utf-8")
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
    # Boundary errors are zero and may be included without changing the sum.
    return math.sqrt(h * sum(value * value for value in error))


def expected_initial_value(x: float, h: float) -> float:
    tol = 16.0 * math.ulp(1.0)
    if abs(x - 0.4) <= tol or abs(x - 0.6) <= tol:
        return 0.5
    if 0.4 < x < 0.6:
        return 1.0
    return 0.0


def main() -> None:
    config = load_config()
    N = int(config["N"])
    dt = float(config["dt"])
    h = 1.0 / N
    expected_r = float(config["expected_r"])
    if abs(dt / (h * h) - expected_r) > float(config["r_abs_tolerance"]):
        raise AssertionError("frozen r mismatch")

    profiles = read_csv(BASE / "results/et003-03/profiles.csv")
    diagnostics = read_csv(BASE / "results/et003-03/diagnostics.csv")
    violation_rows = read_csv(
        BASE / "results/et003-03/monotonicity_violations.csv"
    )

    output_steps = [int(value) for value in config["output_steps"]]
    expected_profile_rows = (N + 1) * len(output_steps)
    if len(profiles) != expected_profile_rows:
        raise AssertionError("profile inventory mismatch")

    method_columns = [
        "semidiscrete_reference",
        "backward_euler",
        "crank_nicolson",
        "rannacher_r1",
    ]

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in profiles:
        grouped.setdefault(int(row["step"]), []).append(row)

    summary = {
        (row["method"], int(row["step"])): row for row in diagnostics
    }
    if len(summary) != len(output_steps) * len(method_columns):
        raise AssertionError("diagnostic inventory mismatch")

    expected_violations: set[tuple[int, str, str, int, int]] = set()

    for step in output_steps:
        rows = grouped.get(step)
        if rows is None or len(rows) != N + 1:
            raise AssertionError(f"step={step}: profile row count")
        rows.sort(key=lambda row: int(row["node_index"]))

        for index, row in enumerate(rows):
            if int(row["node_index"]) != index:
                raise AssertionError(f"step={step}: node index inventory")
            x = float(row["x"])
            if abs(x - index * h) > 2.0e-15:
                raise AssertionError(f"step={step}: x-grid mismatch")

        if step == 0:
            for row in rows:
                x = float(row["x"])
                expected = expected_initial_value(x, h)
                for method in method_columns:
                    if abs(float(row[method]) - expected) > 2.0e-14:
                        raise AssertionError(
                            f"step=0 x={x}: frozen initial projection mismatch"
                        )

        reference = [float(row["semidiscrete_reference"]) for row in rows]

        for method in method_columns:
            values = [float(row[method]) for row in rows]
            errors = [value - ref for value, ref in zip(values, reference)]
            l2 = discrete_l2(errors, h)
            linf = max(abs(value) for value in errors)
            tv = total_variation(values)
            min_value = min(values)
            max_value = max(values)
            violations = monotonicity_violation_edges(
                values,
                tolerance=float(config["monotonicity_tolerance"]),
            )

            stored = summary[(method, step)]
            comparisons = {
                "l2_error_vs_semidiscrete": l2,
                "linf_error_vs_semidiscrete": linf,
                "total_variation": tv,
                "min_value": min_value,
                "max_value": max_value,
            }
            for key, expected in comparisons.items():
                observed = float(stored[key])
                scale = max(abs(expected), 1.0)
                if abs(observed - expected) > 2.0e-13 * scale:
                    raise AssertionError(
                        f"step={step} {method}: stored {key} mismatch"
                    )

            if int(stored["monotonicity_violation_count"]) != len(violations):
                raise AssertionError(
                    f"step={step} {method}: stored violation count mismatch"
                )

            for side, left, right, _ in violations:
                expected_violations.add((step, method, side, left, right))

    stored_violations = {
        (
            int(row["step"]),
            row["method"],
            row["side"],
            int(row["edge_left_index"]),
            int(row["edge_right_index"]),
        )
        for row in violation_rows
    }
    if stored_violations != expected_violations:
        raise AssertionError("monotonicity violation location inventory mismatch")

    for step in output_steps:
        if int(summary[("semidiscrete_reference", step)][
            "monotonicity_violation_count"
        ]) > int(config["max_reference_violation_count"]):
            raise AssertionError(f"reference step={step}: monotonicity")

        if int(summary[("backward_euler", step)][
            "monotonicity_violation_count"
        ]) > int(config["max_be_violation_count"]):
            raise AssertionError(f"BE step={step}: monotonicity")

        if int(summary[("rannacher_r1", step)][
            "monotonicity_violation_count"
        ]) > int(config["max_r1_violation_count"]):
            raise AssertionError(f"R1 step={step}: monotonicity")

    if int(summary[("crank_nicolson", 1)][
        "monotonicity_violation_count"
    ]) < int(config["min_cn_step1_violation_count"]):
        raise AssertionError("CN step=1 did not expose frozen pressure response")

    print("ET003-03 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
