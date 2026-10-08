#!/usr/bin/env python3
"""Independent standard-library checker for ET003-05."""

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
        (BASE / "config/et003-05.json").read_text(encoding="utf-8")
    )


def box_initial_value(x: float) -> float:
    tol = 16.0 * math.ulp(1.0)
    if abs(x - 0.4) <= tol or abs(x - 0.6) <= tol:
        return 0.5
    if 0.4 < x < 0.6:
        return 1.0
    return 0.0


def analytical_eigenvalue(N: int, mode: int) -> float:
    h = 1.0 / N
    return -4.0 / (h * h) * math.sin(mode * math.pi / (2.0 * N)) ** 2


def semidiscrete_reference(N: int, T: float) -> list[float]:
    """Direct discrete-sine expansion, independent of SciPy/DST."""
    scale = math.sqrt(2.0 / N)
    initial = [
        box_initial_value(i / N)
        for i in range(1, N)
    ]

    coefficients: list[float] = []
    for mode in range(1, N):
        total = 0.0
        for i, value in enumerate(initial, start=1):
            total += value * math.sin(i * mode * math.pi / N)
        coefficients.append(scale * total)

    interior: list[float] = []
    for i in range(1, N):
        total = 0.0
        for mode, coefficient in enumerate(coefficients, start=1):
            total += (
                scale
                * coefficient
                * math.exp(analytical_eigenvalue(N, mode) * T)
                * math.sin(i * mode * math.pi / N)
            )
        interior.append(total)

    return [0.0] + interior + [0.0]


def continuous_fourier_reference(
    N: int,
    T: float,
    modes: int,
) -> list[float]:
    values: list[float] = [0.0]
    for i in range(1, N):
        x = i / N
        total = 0.0
        for mode in range(1, modes + 1):
            coefficient = (
                2.0
                / (mode * math.pi)
                * (
                    math.cos(0.4 * mode * math.pi)
                    - math.cos(0.6 * mode * math.pi)
                )
            )
            total += (
                coefficient
                * math.exp(-((mode * math.pi) ** 2) * T)
                * math.sin(mode * math.pi * x)
            )
        values.append(total)
    values.append(0.0)
    return values


def discrete_l2(error: list[float], h: float) -> float:
    return math.sqrt(h * sum(value * value for value in error))


def total_variation(values: list[float]) -> float:
    return sum(abs(b - a) for a, b in zip(values, values[1:]))


def violation_count(values: list[float], tolerance: float) -> int:
    N = len(values) - 1
    count = 0
    for edge in range(N):
        delta = values[edge + 1] - values[edge]
        midpoint = (edge + 0.5) / N
        if midpoint < 0.5 and delta < -tolerance:
            count += 1
        elif midpoint > 0.5 and delta > tolerance:
            count += 1
    return count


def observed_order(previous: float, current: float) -> float:
    return math.log(previous / current) / math.log(2.0)


def main() -> None:
    config = load_config()
    N = int(config["N"])
    T = float(config["T"])
    h = 1.0 / N
    M_values = [int(value) for value in config["M_values"]]
    methods = [str(value) for value in config["methods"]]
    tolerance = float(config["monotonicity_tolerance"])

    convergence = read_csv(BASE / "results/et003-05/convergence.csv")
    profiles = read_csv(BASE / "results/et003-05/profiles.csv")
    audit_rows = read_csv(BASE / "results/et003-05/reference_audit.csv")
    if len(audit_rows) != 1:
        raise AssertionError("ET003-05 reference audit inventory")
    audit = audit_rows[0]

    if len(convergence) != len(M_values) * len(methods):
        raise AssertionError("ET003-05 convergence inventory")
    if len(profiles) != len(M_values) * (N + 1):
        raise AssertionError("ET003-05 profile inventory")

    tight_modes = int(audit["fourier_modes_tight"])
    semidiscrete = semidiscrete_reference(N, T)
    continuous = continuous_fourier_reference(N, T, tight_modes)

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in profiles:
        grouped.setdefault(int(row["M"]), []).append(row)

    convergence_map = {
        (row["method"], int(row["M"])): row for row in convergence
    }

    method_time_errors: dict[str, list[tuple[float, float]]] = {
        method: [] for method in methods
    }

    for M in M_values:
        rows = grouped.get(M)
        if rows is None or len(rows) != N + 1:
            raise AssertionError(f"M={M}: profile inventory")
        rows.sort(key=lambda row: int(row["node_index"]))

        dt = T / M
        r = dt / (h * h)

        for index, row in enumerate(rows):
            if int(row["node_index"]) != index:
                raise AssertionError(f"M={M}: node index")
            if abs(float(row["x"]) - index * h) > 2.0e-15:
                raise AssertionError(f"M={M}: x grid")
            if abs(float(row["dt"]) - dt) > 2.0e-18:
                raise AssertionError(f"M={M}: dt")
            if abs(float(row["r"]) - r) > 2.0e-12:
                raise AssertionError(f"M={M}: r")
            if abs(
                float(row["semidiscrete_reference"]) - semidiscrete[index]
            ) > 4.0e-13:
                raise AssertionError(f"M={M}: semidiscrete reference")
            if abs(
                float(row["continuous_fourier_reference"]) - continuous[index]
            ) > 4.0e-13:
                raise AssertionError(f"M={M}: continuous Fourier reference")

        for method in methods:
            values = [float(row[method]) for row in rows]
            time_error = [
                value - ref for value, ref in zip(values, semidiscrete)
            ]
            pde_error = [
                value - ref for value, ref in zip(values, continuous)
            ]
            l2_time = discrete_l2(time_error, h)
            linf_time = max(abs(value) for value in time_error)
            stored = convergence_map[(method, M)]

            comparisons = {
                "time_l2_error": l2_time,
                "time_linf_error": linf_time,
                "pde_l2_error": discrete_l2(pde_error, h),
                "pde_linf_error": max(abs(value) for value in pde_error),
                "total_variation": total_variation(values),
                "min_value": min(values),
                "max_value": max(values),
            }
            for key, expected in comparisons.items():
                observed = float(stored[key])
                scale = max(abs(expected), 1.0)
                if abs(observed - expected) > 4.0e-13 * scale:
                    raise AssertionError(f"M={M} {method}: {key}")

            expected_violations = violation_count(values, tolerance)
            if int(stored["monotonicity_violation_count"]) != expected_violations:
                raise AssertionError(f"M={M} {method}: violation count")

            expected_solves = M if method != "rannacher_r1" else M + 1
            if int(stored["linear_solves"]) != expected_solves:
                raise AssertionError(f"M={M} {method}: linear solve count")
            expected_extra = 1 if method == "rannacher_r1" else 0
            if int(stored["extra_startup_solves_vs_cn"]) != expected_extra:
                raise AssertionError(f"M={M} {method}: startup solve cost")

            method_time_errors[method].append((l2_time, linf_time))

    for method in methods:
        errors = method_time_errors[method]
        for index, M in enumerate(M_values):
            stored = convergence_map[(method, M)]
            if index == 0:
                if stored["l2_order"] or stored["linf_order"]:
                    raise AssertionError(f"{method}: first order entry must be blank")
                continue

            l2_order = observed_order(errors[index - 1][0], errors[index][0])
            linf_order = observed_order(errors[index - 1][1], errors[index][1])
            if abs(float(stored["l2_order"]) - l2_order) > 1.0e-8:
                raise AssertionError(f"M={M} {method}: L2 order")
            if abs(float(stored["linf_order"]) - linf_order) > 1.0e-8:
                raise AssertionError(f"M={M} {method}: Linf order")

    spatial_error = [
        semi - cont for semi, cont in zip(semidiscrete, continuous)
    ]
    spatial_l2 = discrete_l2(spatial_error, h)
    spatial_linf = max(abs(value) for value in spatial_error)

    if abs(float(audit["spatial_l2_baseline"]) - spatial_l2) > 4.0e-13:
        raise AssertionError("spatial L2 baseline")
    if abs(float(audit["spatial_linf_baseline"]) - spatial_linf) > 4.0e-13:
        raise AssertionError("spatial Linf baseline")
    if float(audit["fourier_refinement_linf"]) > float(
        config["max_fourier_refinement_linf"]
    ):
        raise AssertionError("Fourier refinement gate")

    print("ET003-05 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
