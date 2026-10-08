#!/usr/bin/env python3
"""Run the frozen tutorial-003 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

from heat_core import (
    advance,
    amplification_backward_euler,
    amplification_crank_nicolson,
    amplification_rannacher_startup,
    analytical_eigenvalue,
    analytical_eigenvalues,
    discrete_laplacian,
    make_grid,
    monotonicity_violations,
    sine_mode,
    total_variation,
    with_zero_boundaries,
)
from references import (
    box_initial,
    continuous_box_reference,
    semidiscrete_reference,
    smooth_continuous,
    smooth_initial,
)


# Locate frozen configs/results relative to this source file. This works in
# private experiments/tutorial-003/ and public code/tutorial-003/ alike.
BASE = Path(__file__).resolve().parent.parent


def _load_config(name: str) -> dict:
    return json.loads((BASE / "config" / f"{name}.json").read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _discrete_l2(error: np.ndarray, h: float) -> float:
    return math.sqrt(h * float(np.dot(error, error)))


def _observed_order(previous: float, current: float) -> float:
    return math.log(previous / current) / math.log(2.0)


def _exact_amplification(method: str, xi: float) -> float:
    if method == "backward_euler":
        return float(amplification_backward_euler(xi))
    if method == "crank_nicolson":
        return float(amplification_crank_nicolson(xi))
    if method == "rannacher_r1":
        return float(amplification_rannacher_startup(xi))
    raise ValueError(method)


def run_et003_01() -> None:
    config = _load_config("et003-01")
    result_dir = BASE / "results" / "et003-01"

    spectrum_rows: list[dict[str, object]] = []
    amplification_rows: list[dict[str, object]] = []
    max_spectrum_relative = 0.0
    max_amp_abs = 0.0

    for N_raw in config["N_values"]:
        N = int(N_raw)
        grid = make_grid(N)
        n = N - 1
        inv_h2 = 1.0 / (grid.h * grid.h)

        numeric_ascending = eigh_tridiagonal(
            np.full(n, -2.0 * inv_h2),
            np.full(n - 1, inv_h2),
            eigvals_only=True,
        )
        # scipy returns most-negative first; mode j=1 is the least-negative.
        numeric_by_mode = numeric_ascending[::-1]
        analytical = analytical_eigenvalues(grid)

        for j, (observed, expected) in enumerate(
            zip(numeric_by_mode, analytical),
            start=1,
        ):
            abs_error = abs(float(observed - expected))
            rel_error = abs_error / max(abs(float(expected)), 1.0)
            max_spectrum_relative = max(max_spectrum_relative, rel_error)
            spectrum_rows.append(
                {
                    "N": N,
                    "mode": j,
                    "h": grid.h,
                    "numeric_eigenvalue": float(observed),
                    "analytic_eigenvalue": float(expected),
                    "abs_error": abs_error,
                    "relative_error": rel_error,
                }
            )

        representative_modes = sorted({1, max(1, N // 2), N - 1})
        matrix = discrete_laplacian(grid)
        for r_raw in config["r_values"]:
            r = float(r_raw)
            dt = r * grid.h * grid.h
            for mode in representative_modes:
                vector = sine_mode(grid, mode)
                lam = analytical_eigenvalue(grid, mode)
                xi = -lam * dt

                for method in config["methods"]:
                    result = advance(method, matrix, vector, dt, 1)
                    measured = float(np.dot(vector, result.values))
                    expected = _exact_amplification(method, xi)
                    abs_error = abs(measured - expected)
                    max_amp_abs = max(max_amp_abs, abs_error)
                    amplification_rows.append(
                        {
                            "N": N,
                            "mode": mode,
                            "r": r,
                            "dt": dt,
                            "xi": xi,
                            "method": method,
                            "measured_amplification": measured,
                            "analytic_amplification": expected,
                            "abs_error": abs_error,
                            "linear_solves": result.linear_solves,
                        }
                    )

    spectrum_tol = float(config["max_spectrum_relative_error"])
    amp_tol = float(config["max_amplification_abs_error"])
    if max_spectrum_relative > spectrum_tol:
        raise AssertionError(
            f"spectrum relative error {max_spectrum_relative:.3e} > {spectrum_tol:.3e}"
        )
    if max_amp_abs > amp_tol:
        raise AssertionError(
            f"amplification error {max_amp_abs:.3e} > {amp_tol:.3e}"
        )

    _write_csv(result_dir / "spectrum_check.csv", spectrum_rows)
    _write_csv(result_dir / "amplification_check.csv", amplification_rows)

    print(
        "ET003-01 PASS "
        f"max_spectrum_relative_error={max_spectrum_relative:.3e} "
        f"max_amplification_abs_error={max_amp_abs:.3e}"
    )


def run_et003_02() -> None:
    config = _load_config("et003-02")
    result_dir = BASE / "results" / "et003-02"

    N = int(config["N"])
    T = float(config["T"])
    grid = make_grid(N)
    matrix = discrete_laplacian(grid)
    initial = smooth_initial(grid)

    semidiscrete = semidiscrete_reference(grid, initial, T)
    continuous = smooth_continuous(grid.x, T)

    # The initial data is exactly discrete sine mode j=1.  This gives an
    # independent scalar cross-check of the DST-based semidiscrete reference.
    lam1 = analytical_eigenvalue(grid, 1)
    modal_reference = math.exp(lam1 * T) * initial
    semidiscrete_crosscheck = float(
        np.max(np.abs(semidiscrete - modal_reference))
    )
    max_ref_defect = float(config["max_semidiscrete_reference_crosscheck"])
    if semidiscrete_crosscheck > max_ref_defect:
        raise AssertionError(
            f"semidiscrete reference defect {semidiscrete_crosscheck:.3e} "
            f"> {max_ref_defect:.3e}"
        )

    spatial_error = semidiscrete - continuous
    spatial_l2 = _discrete_l2(spatial_error, grid.h)
    spatial_linf = float(np.max(np.abs(spatial_error)))

    rows: list[dict[str, object]] = []
    per_method_errors: dict[str, list[tuple[float, float]]] = {
        method: [] for method in config["methods"]
    }

    for method in config["methods"]:
        previous_l2: float | None = None
        previous_linf: float | None = None

        for M_raw in config["M_values"]:
            M = int(M_raw)
            dt = T / M
            r = dt / (grid.h * grid.h)
            result = advance(method, matrix, initial, dt, M)

            time_error = result.values - semidiscrete
            pde_error = result.values - continuous
            l2_time = _discrete_l2(time_error, grid.h)
            linf_time = float(np.max(np.abs(time_error)))
            l2_pde = _discrete_l2(pde_error, grid.h)
            linf_pde = float(np.max(np.abs(pde_error)))

            order_l2: str | float = ""
            order_linf: str | float = ""
            if previous_l2 is not None and previous_linf is not None:
                order_l2 = _observed_order(previous_l2, l2_time)
                order_linf = _observed_order(previous_linf, linf_time)

            rows.append(
                {
                    "method": method,
                    "N": N,
                    "M": M,
                    "h": grid.h,
                    "dt": dt,
                    "r": r,
                    "time_l2_error": l2_time,
                    "time_linf_error": linf_time,
                    "l2_order": order_l2,
                    "linf_order": order_linf,
                    "pde_l2_error": l2_pde,
                    "pde_linf_error": linf_pde,
                    "linear_solves": result.linear_solves,
                }
            )
            per_method_errors[method].append((l2_time, linf_time))
            previous_l2 = l2_time
            previous_linf = linf_time

    checks = int(config["check_last_n_orders"])
    for method in config["methods"]:
        errors = per_method_errors[method]
        if not all(a[0] > b[0] > 0.0 for a, b in zip(errors, errors[1:])):
            raise AssertionError(f"{method}: L2 time error is not strictly decreasing")
        if not all(a[1] > b[1] > 0.0 for a, b in zip(errors, errors[1:])):
            raise AssertionError(f"{method}: Linf time error is not strictly decreasing")

        l2_orders = [
            _observed_order(a[0], b[0]) for a, b in zip(errors, errors[1:])
        ]
        linf_orders = [
            _observed_order(a[1], b[1]) for a, b in zip(errors, errors[1:])
        ]
        low, high = (float(value) for value in config["order_windows"][method])
        for value in l2_orders[-checks:] + linf_orders[-checks:]:
            if not low <= value <= high:
                raise AssertionError(
                    f"{method}: observed order {value:.8f} outside [{low}, {high}]"
                )

    _write_csv(result_dir / "convergence.csv", rows)
    _write_csv(
        result_dir / "reference_audit.csv",
        [
            {
                "N": N,
                "T": T,
                "semidiscrete_modal_crosscheck_linf": semidiscrete_crosscheck,
                "spatial_l2_baseline": spatial_l2,
                "spatial_linf_baseline": spatial_linf,
            }
        ],
    )

    print(
        "ET003-02 PASS "
        f"semidiscrete_crosscheck={semidiscrete_crosscheck:.3e} "
        f"spatial_l2_baseline={spatial_l2:.16e} "
        f"spatial_linf_baseline={spatial_linf:.16e}"
    )
    for method in config["methods"]:
        method_rows = [row for row in rows if row["method"] == method]
        tail = method_rows[-1]
        print(
            f"{method}: M={tail['M']} "
            f"L2_time={float(tail['time_l2_error']):.16e} "
            f"L2_order={float(tail['l2_order']):.8f} "
            f"Linf_time={float(tail['time_linf_error']):.16e} "
            f"Linf_order={float(tail['linf_order']):.8f} "
            f"linear_solves={tail['linear_solves']}"
        )


def run_et003_03() -> None:
    config = _load_config("et003-03")
    result_dir = BASE / "results" / "et003-03"

    N = int(config["N"])
    dt = float(config["dt"])
    output_steps = [int(value) for value in config["output_steps"]]
    methods = [str(value) for value in config["methods"]]
    tolerance = float(config["monotonicity_tolerance"])

    grid = make_grid(N)
    matrix = discrete_laplacian(grid)
    initial = box_initial(grid)
    r = dt / (grid.h * grid.h)

    expected_r = float(config["expected_r"])
    if abs(r - expected_r) > float(config["r_abs_tolerance"]):
        raise AssertionError(f"computed r={r} does not match frozen r={expected_r}")

    profile_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []

    x_full = np.linspace(0.0, 1.0, N + 1)

    for step in output_steps:
        t = step * dt
        if step == 0:
            reference = initial.copy()
        else:
            reference = semidiscrete_reference(grid, initial, t)

        states: dict[str, tuple[np.ndarray, int]] = {
            "semidiscrete_reference": (reference, 0)
        }

        for method in methods:
            if step == 0:
                states[method] = (initial.copy(), 0)
            else:
                result = advance(method, matrix, initial, dt, step)
                states[method] = (result.values, result.linear_solves)

        full_states = {
            name: with_zero_boundaries(grid, values)
            for name, (values, _) in states.items()
        }

        for node_index, x in enumerate(x_full):
            profile_rows.append(
                {
                    "step": step,
                    "time": t,
                    "node_index": node_index,
                    "x": float(x),
                    "semidiscrete_reference": float(
                        full_states["semidiscrete_reference"][node_index]
                    ),
                    "backward_euler": float(full_states["backward_euler"][node_index]),
                    "crank_nicolson": float(
                        full_states["crank_nicolson"][node_index]
                    ),
                    "rannacher_r1": float(full_states["rannacher_r1"][node_index]),
                }
            )

        for method_name, (values, linear_solves) in states.items():
            error = values - reference
            l2_error = _discrete_l2(error, grid.h)
            linf_error = float(np.max(np.abs(error)))
            violations = monotonicity_violations(
                grid,
                values,
                tolerance=tolerance,
            )

            full = full_states[method_name]
            if not np.all(np.isfinite(full)):
                raise AssertionError(f"{method_name} step={step}: non-finite values")

            diagnostic_rows.append(
                {
                    "step": step,
                    "time": t,
                    "method": method_name,
                    "N": N,
                    "h": grid.h,
                    "dt": dt,
                    "r": r,
                    "l2_error_vs_semidiscrete": l2_error,
                    "linf_error_vs_semidiscrete": linf_error,
                    "total_variation": total_variation(grid, values),
                    "min_value": float(np.min(full)),
                    "max_value": float(np.max(full)),
                    "monotonicity_violation_count": len(violations),
                    "linear_solves": linear_solves,
                }
            )

            for item in violations:
                violation_rows.append(
                    {
                        "step": step,
                        "time": t,
                        "method": method_name,
                        **item,
                    }
                )

    summary_by_key = {
        (str(row["method"]), int(row["step"])): row for row in diagnostic_rows
    }

    for step in output_steps:
        reference_count = int(
            summary_by_key[("semidiscrete_reference", step)][
                "monotonicity_violation_count"
            ]
        )
        if reference_count > int(config["max_reference_violation_count"]):
            raise AssertionError(
                f"reference step={step}: monotonicity violations={reference_count}"
            )

        for method, config_key in (
            ("backward_euler", "max_be_violation_count"),
            ("rannacher_r1", "max_r1_violation_count"),
        ):
            count = int(
                summary_by_key[(method, step)]["monotonicity_violation_count"]
            )
            if count > int(config[config_key]):
                raise AssertionError(
                    f"{method} step={step}: monotonicity violations={count}"
                )

    cn_step1 = int(
        summary_by_key[("crank_nicolson", 1)]["monotonicity_violation_count"]
    )
    if cn_step1 < int(config["min_cn_step1_violation_count"]):
        raise AssertionError(
            f"CN step=1 violations={cn_step1}; frozen pressure case did not expose "
            "the expected non-monotone response"
        )

    _write_csv(result_dir / "profiles.csv", profile_rows)
    _write_csv(result_dir / "diagnostics.csv", diagnostic_rows)

    # Keep a valid CSV even if only some methods have violations.
    violation_path = result_dir / "monotonicity_violations.csv"
    violation_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "time",
        "method",
        "side",
        "edge_left_index",
        "edge_right_index",
        "x_left",
        "x_right",
        "delta",
    ]
    with violation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(violation_rows)

    print(
        "ET003-03 PASS "
        f"N={N} h={grid.h:.16e} dt={dt:.16e} r={r:.8f}"
    )
    for step in output_steps:
        print(f"step={step} t={step * dt:.8e}")
        for method_name in (
            "semidiscrete_reference",
            "backward_euler",
            "crank_nicolson",
            "rannacher_r1",
        ):
            row = summary_by_key[(method_name, step)]
            print(
                f"  {method_name}: "
                f"L2={float(row['l2_error_vs_semidiscrete']):.8e} "
                f"Linf={float(row['linf_error_vs_semidiscrete']):.8e} "
                f"TV={float(row['total_variation']):.8e} "
                f"min={float(row['min_value']):.8e} "
                f"max={float(row['max_value']):.8e} "
                f"violations={row['monotonicity_violation_count']} "
                f"linear_solves={row['linear_solves']}"
            )


def run_et003_04() -> None:
    config = _load_config("et003-04")
    result_dir = BASE / "results" / "et003-04"

    N = int(config["N"])
    grid = make_grid(N)
    matrix = discrete_laplacian(grid)
    initial = box_initial(grid)
    methods = [str(value) for value in config["methods"]]
    tolerance = float(config["monotonicity_tolerance"])
    r_values = [float(value) for value in config["r_values"]]

    r_flip = 1.0 / (
        2.0 * math.cos(math.pi / (2.0 * N)) ** 2
    )
    xi_max_factor = 4.0 * math.cos(math.pi / (2.0 * N)) ** 2

    scan_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []

    x_full = np.linspace(0.0, 1.0, N + 1)

    for r in r_values:
        dt = r * grid.h * grid.h
        reference = semidiscrete_reference(grid, initial, dt)

        states: dict[str, tuple[np.ndarray, int]] = {
            "semidiscrete_reference": (reference, 0)
        }
        for method in methods:
            result = advance(method, matrix, initial, dt, 1)
            states[method] = (result.values, result.linear_solves)

        full_states = {
            name: with_zero_boundaries(grid, values)
            for name, (values, _) in states.items()
        }

        xi_max = xi_max_factor * r
        g_cn_max_mode = float(amplification_crank_nicolson(xi_max))

        for node_index, x in enumerate(x_full):
            profile_rows.append(
                {
                    "r": r,
                    "dt": dt,
                    "node_index": node_index,
                    "x": float(x),
                    "semidiscrete_reference": float(
                        full_states["semidiscrete_reference"][node_index]
                    ),
                    "backward_euler": float(full_states["backward_euler"][node_index]),
                    "crank_nicolson": float(
                        full_states["crank_nicolson"][node_index]
                    ),
                    "rannacher_r1": float(full_states["rannacher_r1"][node_index]),
                }
            )

        for method_name, (values, linear_solves) in states.items():
            error = values - reference
            violations = monotonicity_violations(
                grid,
                values,
                tolerance=tolerance,
            )
            full = full_states[method_name]

            scan_rows.append(
                {
                    "r": r,
                    "dt": dt,
                    "N": N,
                    "h": grid.h,
                    "r_flip": r_flip,
                    "xi_max": xi_max,
                    "cn_highest_mode_amplification": g_cn_max_mode,
                    "highest_mode_sign_flipped": int(g_cn_max_mode < 0.0),
                    "method": method_name,
                    "l2_error_vs_semidiscrete": _discrete_l2(error, grid.h),
                    "linf_error_vs_semidiscrete": float(
                        np.max(np.abs(error))
                    ),
                    "total_variation": total_variation(grid, values),
                    "min_value": float(np.min(full)),
                    "max_value": float(np.max(full)),
                    "monotonicity_violation_count": len(violations),
                    "linear_solves": linear_solves,
                }
            )

            for item in violations:
                violation_rows.append(
                    {
                        "r": r,
                        "dt": dt,
                        "method": method_name,
                        **item,
                    }
                )

    summary = {
        (str(row["method"]), float(row["r"])): row for row in scan_rows
    }

    for r in r_values:
        for method, config_key in (
            ("semidiscrete_reference", "max_reference_violation_count"),
            ("backward_euler", "max_be_violation_count"),
            ("rannacher_r1", "max_r1_violation_count"),
        ):
            count = int(summary[(method, r)]["monotonicity_violation_count"])
            if count > int(config[config_key]):
                raise AssertionError(
                    f"{method} r={r}: monotonicity violations={count}"
                )

    pressure_r = float(config["pressure_r"])
    if pressure_r not in r_values:
        raise AssertionError("pressure_r must be one of the frozen r values")
    pressure_count = int(
        summary[("crank_nicolson", pressure_r)][
            "monotonicity_violation_count"
        ]
    )
    if pressure_count < int(config["min_cn_pressure_violation_count"]):
        raise AssertionError(
            f"CN r={pressure_r}: violations={pressure_count}; "
            "frozen scan lost the pressure-test response"
        )

    _write_csv(result_dir / "scan.csv", scan_rows)
    _write_csv(result_dir / "profiles.csv", profile_rows)

    violation_path = result_dir / "monotonicity_violations.csv"
    violation_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "r",
        "dt",
        "method",
        "side",
        "edge_left_index",
        "edge_right_index",
        "x_left",
        "x_right",
        "delta",
    ]
    with violation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(violation_rows)

    cn_rows = [
        summary[("crank_nicolson", r)] for r in r_values
    ]
    negative_r = [
        r for r in r_values
        if int(summary[("crank_nicolson", r)]["highest_mode_sign_flipped"]) == 1
    ]
    visible_r = [
        r for r in r_values
        if int(summary[("crank_nicolson", r)]["monotonicity_violation_count"]) > 0
    ]

    print(
        "ET003-04 PASS "
        f"N={N} h={grid.h:.16e} r_flip={r_flip:.12f} "
        f"first_sampled_negative_highest_mode="
        f"{negative_r[0] if negative_r else 'none'} "
        f"first_sampled_visible_violation="
        f"{visible_r[0] if visible_r else 'none'}"
    )
    for row in cn_rows:
        print(
            f"r={float(row['r']):>6g} "
            f"xi_max={float(row['xi_max']):.8f} "
            f"Gmax={float(row['cn_highest_mode_amplification']): .8f} "
            f"flip={row['highest_mode_sign_flipped']} "
            f"TV={float(row['total_variation']):.8f} "
            f"Linf={float(row['linf_error_vs_semidiscrete']):.8e} "
            f"violations={row['monotonicity_violation_count']} "
            f"min={float(row['min_value']):.8e} "
            f"max={float(row['max_value']):.8e}"
        )


def run_et003_05() -> None:
    config = _load_config("et003-05")
    result_dir = BASE / "results" / "et003-05"

    N = int(config["N"])
    T = float(config["T"])
    M_values = [int(value) for value in config["M_values"]]
    methods = [str(value) for value in config["methods"]]
    tolerance = float(config["monotonicity_tolerance"])

    grid = make_grid(N)
    matrix = discrete_laplacian(grid)
    initial = box_initial(grid)

    semidiscrete = semidiscrete_reference(grid, initial, T)
    continuous, fourier_modes = continuous_box_reference(
        grid.x,
        T,
        tolerance=float(config["fourier_tolerance"]),
        max_modes=int(config["fourier_max_modes"]),
    )
    continuous_tight, fourier_modes_tight = continuous_box_reference(
        grid.x,
        T,
        tolerance=float(config["fourier_tight_tolerance"]),
        max_modes=int(config["fourier_max_modes"]),
    )
    fourier_refinement_linf = float(
        np.max(np.abs(continuous - continuous_tight))
    )
    if fourier_refinement_linf > float(config["max_fourier_refinement_linf"]):
        raise AssertionError(
            "continuous Fourier reference refinement defect "
            f"{fourier_refinement_linf:.3e} exceeds frozen gate"
        )

    # Use the tighter series as the authoritative continuous-PDE reference.
    continuous = continuous_tight

    spatial_error = semidiscrete - continuous
    spatial_l2 = _discrete_l2(spatial_error, grid.h)
    spatial_linf = float(np.max(np.abs(spatial_error)))

    reference_full = with_zero_boundaries(grid, semidiscrete)
    continuous_full = with_zero_boundaries(grid, continuous)
    reference_tv = total_variation(grid, semidiscrete)
    continuous_tv = total_variation(grid, continuous)

    rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    per_method_errors: dict[str, list[tuple[float, float]]] = {
        method: [] for method in methods
    }

    x_full = np.linspace(0.0, 1.0, N + 1)

    for M in M_values:
        dt = T / M
        r = dt / (grid.h * grid.h)

        states: dict[str, tuple[np.ndarray, int]] = {}
        for method in methods:
            result = advance(method, matrix, initial, dt, M)
            states[method] = (result.values, result.linear_solves)

        full_states = {
            method: with_zero_boundaries(grid, values)
            for method, (values, _) in states.items()
        }

        for node_index, x in enumerate(x_full):
            profile_rows.append(
                {
                    "M": M,
                    "dt": dt,
                    "r": r,
                    "node_index": node_index,
                    "x": float(x),
                    "semidiscrete_reference": float(reference_full[node_index]),
                    "continuous_fourier_reference": float(
                        continuous_full[node_index]
                    ),
                    "backward_euler": float(
                        full_states["backward_euler"][node_index]
                    ),
                    "crank_nicolson": float(
                        full_states["crank_nicolson"][node_index]
                    ),
                    "rannacher_r1": float(
                        full_states["rannacher_r1"][node_index]
                    ),
                }
            )

        for method in methods:
            values, linear_solves = states[method]
            time_error = values - semidiscrete
            pde_error = values - continuous
            violations = monotonicity_violations(
                grid,
                values,
                tolerance=tolerance,
            )

            l2_time = _discrete_l2(time_error, grid.h)
            linf_time = float(np.max(np.abs(time_error)))
            l2_pde = _discrete_l2(pde_error, grid.h)
            linf_pde = float(np.max(np.abs(pde_error)))
            full = full_states[method]

            rows.append(
                {
                    "method": method,
                    "N": N,
                    "T": T,
                    "M": M,
                    "h": grid.h,
                    "dt": dt,
                    "r": r,
                    "time_l2_error": l2_time,
                    "time_linf_error": linf_time,
                    "l2_order": "",
                    "linf_order": "",
                    "pde_l2_error": l2_pde,
                    "pde_linf_error": linf_pde,
                    "total_variation": total_variation(grid, values),
                    "min_value": float(np.min(full)),
                    "max_value": float(np.max(full)),
                    "monotonicity_violation_count": len(violations),
                    "linear_solves": linear_solves,
                    "extra_startup_solves_vs_cn": linear_solves - M,
                }
            )
            per_method_errors[method].append((l2_time, linf_time))

    # Fill adjacent-level observed orders only after all errors are frozen.
    row_lookup = {
        (str(row["method"]), int(row["M"])): row for row in rows
    }
    for method in methods:
        errors = per_method_errors[method]
        for index in range(1, len(M_values)):
            previous_l2, previous_linf = errors[index - 1]
            current_l2, current_linf = errors[index]
            row = row_lookup[(method, M_values[index])]
            row["l2_order"] = _observed_order(previous_l2, current_l2)
            row["linf_order"] = _observed_order(previous_linf, current_linf)

    _write_csv(result_dir / "convergence.csv", rows)
    _write_csv(result_dir / "profiles.csv", profile_rows)
    _write_csv(
        result_dir / "reference_audit.csv",
        [
            {
                "N": N,
                "T": T,
                "fourier_tolerance": float(config["fourier_tolerance"]),
                "fourier_modes": fourier_modes,
                "fourier_tight_tolerance": float(
                    config["fourier_tight_tolerance"]
                ),
                "fourier_modes_tight": fourier_modes_tight,
                "fourier_refinement_linf": fourier_refinement_linf,
                "spatial_l2_baseline": spatial_l2,
                "spatial_linf_baseline": spatial_linf,
                "semidiscrete_total_variation": reference_tv,
                "continuous_total_variation": continuous_tv,
            }
        ],
    )

    print(
        "ET003-05 PASS "
        f"N={N} T={T:.8e} "
        f"Fourier_modes={fourier_modes}/{fourier_modes_tight} "
        f"Fourier_refinement_Linf={fourier_refinement_linf:.3e} "
        f"spatial_L2={spatial_l2:.8e} "
        f"spatial_Linf={spatial_linf:.8e}"
    )
    for method in methods:
        print(method)
        for M in M_values:
            row = row_lookup[(method, M)]
            print(
                f"  M={M:>4} r={float(row['r']):>8.3f} "
                f"L2={float(row['time_l2_error']):.8e} "
                f"p2={row['l2_order']} "
                f"Linf={float(row['time_linf_error']):.8e} "
                f"pinf={row['linf_order']} "
                f"TV={float(row['total_variation']):.8f} "
                f"violations={row['monotonicity_violation_count']} "
                f"solves={row['linear_solves']} "
                f"extra={row['extra_startup_solves_vs_cn']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("et003-01", "et003-02", "et003-03", "et003-04", "et003-05"),
        default=("et003-01", "et003-02"),
    )
    args = parser.parse_args()

    for name in args.experiments:
        if name == "et003-01":
            run_et003_01()
        elif name == "et003-02":
            run_et003_02()
        elif name == "et003-03":
            run_et003_03()
        elif name == "et003-04":
            run_et003_04()
        elif name == "et003-05":
            run_et003_05()

    print("tutorial-003 requested experiments: PASS")


if __name__ == "__main__":
    main()
