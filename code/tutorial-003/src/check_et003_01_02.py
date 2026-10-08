#!/usr/bin/env python3
"""Independent standard-library checker for ET003-01 and ET003-02 outputs."""

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


def load_config(name: str) -> dict:
    return json.loads((BASE / "config" / f"{name}.json").read_text(encoding="utf-8"))


def analytical_eigenvalue(N: int, mode: int) -> float:
    h = 1.0 / N
    return -4.0 / (h * h) * math.sin(mode * math.pi / (2.0 * N)) ** 2


def exact_amplification(method: str, xi: float) -> float:
    if method == "backward_euler":
        return 1.0 / (1.0 + xi)
    if method == "crank_nicolson":
        return (1.0 - 0.5 * xi) / (1.0 + 0.5 * xi)
    if method == "rannacher_r1":
        return 1.0 / (1.0 + 0.5 * xi) ** 2
    raise AssertionError(f"unexpected method {method}")


def observed_order(previous: float, current: float) -> float:
    return math.log(previous / current) / math.log(2.0)


def check_et003_01() -> None:
    config = load_config("et003-01")
    spectrum = read_csv(BASE / "results/et003-01/spectrum_check.csv")
    amplification = read_csv(BASE / "results/et003-01/amplification_check.csv")

    expected_spectrum_rows = sum(int(N) - 1 for N in config["N_values"])
    if len(spectrum) != expected_spectrum_rows:
        raise AssertionError("ET003-01 spectrum inventory mismatch")

    spectrum_tol = float(config["max_spectrum_relative_error"])
    for row in spectrum:
        N = int(row["N"])
        mode = int(row["mode"])
        expected = analytical_eigenvalue(N, mode)
        stored_analytic = float(row["analytic_eigenvalue"])
        numeric = float(row["numeric_eigenvalue"])
        if abs(stored_analytic - expected) > 1.0e-13 * max(abs(expected), 1.0):
            raise AssertionError(f"N={N} mode={mode}: stored analytical eigenvalue mismatch")
        relative = abs(numeric - expected) / max(abs(expected), 1.0)
        if relative > spectrum_tol:
            raise AssertionError(f"N={N} mode={mode}: spectrum tolerance")

    expected_amp_rows = (
        len(config["N_values"])
        * len(config["r_values"])
        * 3
        * len(config["methods"])
    )
    if len(amplification) != expected_amp_rows:
        raise AssertionError(
            f"ET003-01 amplification inventory {len(amplification)} != {expected_amp_rows}"
        )

    amp_tol = float(config["max_amplification_abs_error"])
    for row in amplification:
        xi = float(row["xi"])
        method = row["method"]
        expected = exact_amplification(method, xi)
        stored = float(row["analytic_amplification"])
        measured = float(row["measured_amplification"])
        if abs(stored - expected) > 1.0e-14:
            raise AssertionError("stored amplification formula mismatch")
        if abs(measured - expected) > amp_tol:
            raise AssertionError(
                f"N={row['N']} mode={row['mode']} r={row['r']} {method}: amplification"
            )


def check_et003_02() -> None:
    config = load_config("et003-02")
    rows = read_csv(BASE / "results/et003-02/convergence.csv")
    audit = read_csv(BASE / "results/et003-02/reference_audit.csv")

    expected_rows = len(config["methods"]) * len(config["M_values"])
    if len(rows) != expected_rows:
        raise AssertionError("ET003-02 convergence inventory mismatch")
    if len(audit) != 1:
        raise AssertionError("ET003-02 reference audit inventory mismatch")

    if float(audit[0]["semidiscrete_modal_crosscheck_linf"]) > float(
        config["max_semidiscrete_reference_crosscheck"]
    ):
        raise AssertionError("semidiscrete reference cross-check failed")

    check_last = int(config["check_last_n_orders"])
    for method in config["methods"]:
        method_rows = [row for row in rows if row["method"] == method]
        if [int(row["M"]) for row in method_rows] != [
            int(M) for M in config["M_values"]
        ]:
            raise AssertionError(f"{method}: M inventory mismatch")

        l2 = [float(row["time_l2_error"]) for row in method_rows]
        linf = [float(row["time_linf_error"]) for row in method_rows]
        if not all(a > b > 0.0 for a, b in zip(l2, l2[1:])):
            raise AssertionError(f"{method}: L2 errors not strictly decreasing")
        if not all(a > b > 0.0 for a, b in zip(linf, linf[1:])):
            raise AssertionError(f"{method}: Linf errors not strictly decreasing")

        l2_orders = [observed_order(a, b) for a, b in zip(l2, l2[1:])]
        linf_orders = [observed_order(a, b) for a, b in zip(linf, linf[1:])]
        low, high = (float(value) for value in config["order_windows"][method])
        for value in l2_orders[-check_last:] + linf_orders[-check_last:]:
            if not low <= value <= high:
                raise AssertionError(
                    f"{method}: recomputed order {value:.8f} outside [{low}, {high}]"
                )

        # Recompute the stored order columns rather than trusting the driver.
        for index, row in enumerate(method_rows):
            if index == 0:
                if row["l2_order"] or row["linf_order"]:
                    raise AssertionError(f"{method}: first row should have blank orders")
                continue
            if abs(float(row["l2_order"]) - l2_orders[index - 1]) > 1.0e-12:
                raise AssertionError(f"{method}: stored L2 order mismatch")
            if abs(float(row["linf_order"]) - linf_orders[index - 1]) > 1.0e-12:
                raise AssertionError(f"{method}: stored Linf order mismatch")


def main() -> None:
    check_et003_01()
    check_et003_02()
    print("ET003-01/02 independent CSV checker: PASS")


if __name__ == "__main__":
    main()
