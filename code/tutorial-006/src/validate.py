#!/usr/bin/env python3
"""Independent replay and mathematical checks for tutorial-006.

This file deliberately imports neither wave.py nor run_experiments.py.  It reads
the frozen design and transparent outputs, reconstructs the initial data from
the mathematical definition, and replays all runs using numpy.diff.  A second,
plain-list implementation independently replays all five N=200 cases.

Numerical agreement tolerances below cover floating-point accumulation only.
They are NOT acceptance thresholds for physical reflection, discretization
error, or observed convergence order.  Those quantities remain report data.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np


EPS = np.finfo(float).eps
SNAPSHOT_FIELDS = ["step", "time", "index", "x", "u"]
DIAGNOSTIC_FIELDS = ["step", "time", "min_u", "max_u", "l2_displacement"]
METRIC_FIELDS = [
    "run_id", "N", "boundary", "beta", "h", "dt", "step", "time",
    "l2_continuous_error", "linf_continuous_error",
    "l2_short_large_error", "linf_short_large_error",
    "l2_large_continuous_error", "linf_large_continuous_error",
    "l2_boundary_exact_error", "linf_boundary_exact_error",
    "reflection_target", "reflected_projection", "reflected_fit_l2",
    "reflected_template_l2",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, tolerance, name):
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    require(a.shape == b.shape, name + ": shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), name + ": nonfinite")
    discrepancy = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(discrepancy <= tolerance,
            f"{name}: discrepancy {discrepancy:.17g} exceeds {tolerance:.17g}")
    return discrepancy


def table(path, expected_fields):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == expected_fields,
                str(path) + ": unexpected CSV columns")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            str(path) + ": malformed CSV row")
    return rows


def integer(value, name):
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), name + ": not integer")
    return int(number)


def pulse(x, center, half_width, power):
    """Definition evaluated independently; zero outside the compact support."""
    z = (np.asarray(x, dtype=float) - center) / half_width
    values = np.zeros_like(z)
    inside = np.abs(z) < 1.0
    values[inside] = (1.0 - z[inside] ** 2) ** power
    return values


def pulse_and_velocity_scalar(x, c, center, half_width, power):
    z = (x - center) / half_width
    if abs(z) >= 1.0:
        return 0.0, 0.0
    base = 1.0 - z * z
    return base ** power, c * 2.0 * power * z * base ** (power - 1) / half_width


def norm(values, h):
    a = np.asarray(values, dtype=float)
    weighted = float(np.sum(a[1:-1] ** 2) + 0.5 * (a[0] ** 2 + a[-1] ** 2))
    return math.sqrt(max(0.0, h * weighted))


def inner(a, b, h):
    products = np.asarray(a) * np.asarray(b)
    return h * float(np.sum(products[1:-1]) + 0.5 * (products[0] + products[-1]))


def continuous_reflection(beta):
    return -1.0 if beta is None else (beta - 1.0) / (beta + 1.0)


def mathematical_tests():
    """Exact rational tests of signs, derivatives, and endpoint algebra."""
    tests = []
    for z in [Fraction(-3, 4), Fraction(-1, 2), Fraction(0), Fraction(1, 2),
              Fraction(3, 4)]:
        coefficients = [(-1) ** k * math.comb(6, k) for k in range(7)]
        expanded = sum(Fraction(coefficients[k]) * z ** (2 * k) for k in range(7))
        derivative = sum(Fraction(2 * k * coefficients[k]) * z ** (2 * k - 1)
                         for k in range(1, 7))
        require(expanded == (1 - z * z) ** 6, "pulse polynomial identity")
        require(derivative == -12 * z * (1 - z * z) ** 5,
                "initial velocity derivative identity")
    tests.append("compact-pulse polynomial and first derivative: exact fractions")

    for beta in [Fraction(1, 2), Fraction(1), Fraction(2)]:
        reflection = (beta - 1) / (beta + 1)
        # At x=L, u_t/c=-(1+R)g', u_x=(1-R)g'.
        require(-(1 + reflection) + beta * (1 - reflection) == 0,
                "right outgoing derivative sign")
        require(reflection == {Fraction(1, 2): Fraction(-1, 3),
                               Fraction(1): Fraction(0),
                               Fraction(2): Fraction(1, 3)}[beta],
                "continuous signed reflection coefficient")
        courant_b = beta * Fraction(4, 5)
        q = (courant_b - 1) / (courant_b + 1)
        require(abs(q) < 1, "Mur q magnitude for positive frozen speed")
        for old_boundary, old_inner, new_inner in [
                (Fraction(2, 7), Fraction(-3, 5), Fraction(11, 13)),
                (Fraction(1), Fraction(1), Fraction(1))]:
            new_boundary = old_inner + q * (new_inner - old_boundary)
            # Midpoint discretization of right u_t+c_b*u_x=0, h=1.
            residual = (new_boundary + new_inner - old_boundary - old_inner
                        + courant_b * (new_boundary - new_inner
                                      + old_boundary - old_inner))
            require(residual == 0, "Mur midpoint equation")
    tests.append("right outgoing sign, R(beta), and Mur midpoint equation: exact fractions")
    for speed in [Fraction(1, 2), Fraction(1), Fraction(2)]:
        coordinate, time = Fraction(2, 7), Fraction(3, 11)
        argument = coordinate + speed * time
        profile_derivative = 3 * argument ** 2 - 2
        time_derivative = speed * profile_derivative
        space_derivative = profile_derivative
        require(time_derivative - speed * space_derivative == 0,
                "left-moving cubic profile satisfies left outgoing condition")
        require(time_derivative + speed * space_derivative != 0,
                "incorrect plus sign would fail this left-moving profile")
    tests.append("left outgoing sign: cubic F(x+c*t), exact fractions")

    values = [Fraction(1, 7), Fraction(-3, 2), Fraction(2, 5), Fraction(8, 9),
              Fraction(-1, 3)]
    for j in range(1, len(values) - 1):
        previous = values[j + 1]
        next_value = (2 * values[j] - previous
                      + values[j + 1] - 2 * values[j] + values[j - 1])
        require(next_value == values[j - 1], "C=1 exact lattice shift")
    tests.append("C=1 exact shift is a special algebraic case, not a production run")

    require(1 + continuous_reflection(None) == 0,
            "Dirichlet mirror coefficient")
    tests.append("Dirichlet mirror cancellation: R=-1")
    return tests


def check_design(config):
    expected_model = {
        "c": 1.0,
        "pulse": {"center": 0.35, "half_width": 0.1, "power": 6},
        "short_domain": [0, 1],
        "large_domain": [-2, 3],
        "final_time": 1.2,
    }
    require(config["schema_version"] == 1, "schema version")
    require(config["article_id"] == "tutorial-006", "article id")
    # Narrative labels may coexist with frozen scientific inputs.
    # Check each input explicitly; added documentation is not a model change.
    for field, expected in expected_model.items():
        if field == "pulse":
            for parameter, value in expected.items():
                require(config["model"]["pulse"][parameter] == value,
                        "frozen pulse." + parameter + " differs")
        else:
            require(config["model"][field] == expected,
                    "frozen model." + field + " differs")
    require(config["model"]["equation"] == "u_tt = c^2 u_xx",
            "frozen equation declaration differs")
    require(config["model"]["initial_displacement"] == "g(x)",
            "frozen displacement declaration differs")
    require(config["model"]["initial_velocity"] == "-c*g'(x)",
            "frozen velocity declaration differs")
    d = config["discretization"]
    require(d["courant"] == 0.8, "frozen Courant number differs")
    require(d["interval_counts"] == [200, 400, 800, 1600], "frozen grids differ")
    require(d["snapshot_times"] == [0, 0.4, 0.6, 0.9, 1.2], "frozen times differ")
    expected_boundaries = [
        {"name": "dirichlet", "kind": "dirichlet", "speed_ratio": None},
        {"name": "mur-beta-0p5", "kind": "mur", "speed_ratio": 0.5},
        {"name": "mur-beta-1", "kind": "mur", "speed_ratio": 1.0},
        {"name": "mur-beta-2", "kind": "mur", "speed_ratio": 2.0},
    ]
    require(config["boundaries"] == expected_boundaries, "frozen boundary cases differ")
    require(expected_model["final_time"] < 2 - (0.35 + 0.1),
            "one-right-reflection exact solution has expired")


def replay(config, n_intervals, boundary, large=False, scalar=False):
    """Independently reconstruct every time level and requested snapshots."""
    m = config["model"]
    c = float(m["c"])
    center = float(m["pulse"]["center"])
    width = float(m["pulse"]["half_width"])
    power = int(m["pulse"]["power"])
    h = 1.0 / n_intervals
    dt = config["discretization"]["courant"] * h / c
    steps = int(round(m["final_time"] / dt))
    wanted = {int(round(t / dt)) for t in config["discretization"]["snapshot_times"]}
    for t in config["discretization"]["snapshot_times"] + [m["final_time"]]:
        require(abs(round(t / dt) * dt - t) <= 8 * EPS, "nonintegral time step")
    start, stop = (-2 * n_intervals, 3 * n_intervals) if large else (0, n_intervals)
    x = np.arange(start, stop + 1, dtype=float) / n_intervals
    crop_start = 2 * n_intervals if large else 0
    crop = slice(crop_start, crop_start + n_intervals + 1)
    if large:
        support = np.flatnonzero(pulse(x, center, width, power) != 0)
        require(int(support[0]) > steps and len(x) - 1 - int(support[-1]) > steps,
                "expanded boundary lies in finite stencil domain of dependence")
    beta = None if large else boundary["speed_ratio"]
    mur = not large and boundary["kind"] == "mur"
    beta_courant = beta * c * dt / h if mur else None
    coefficient = (c * dt / h) ** 2

    def finish_endpoints(new, current):
        if not mur:
            new[0], new[-1] = 0.0, 0.0
        else:
            # Solve midpoint equations directly, rather than reuse q-update.
            for edge, adjacent in [(0, 1), (-1, -2)]:
                new[edge] = (
                    (1 - beta_courant) * current[edge]
                    + (1 + beta_courant) * current[adjacent]
                    + (beta_courant - 1) * new[adjacent]
                ) / (1 + beta_courant)

    if scalar:
        initial_pairs = [pulse_and_velocity_scalar(float(pos), c, center, width, power)
                         for pos in x]
        previous = [pair[0] for pair in initial_pairs]
        velocity = [pair[1] for pair in initial_pairs]
        current = [0.0] * len(x)
        for j in range(1, len(x) - 1):
            current[j] = (previous[j] + dt * velocity[j]
                          + coefficient * (previous[j-1] - 2 * previous[j]
                                           + previous[j+1]) / 2)
    else:
        previous = pulse(x, center, width, power)
        z = (x - center) / width
        velocity = np.zeros_like(x)
        inside = np.abs(z) < 1
        velocity[inside] = 2 * c * power * z[inside] * (1 - z[inside] ** 2) ** (power - 1) / width
        current = np.zeros_like(x)
        current[1:-1] = previous[1:-1] + dt * velocity[1:-1] + 0.5 * coefficient * np.diff(previous, n=2)
    finish_endpoints(current, previous)
    snapshots = {}
    diagnostics = []

    def capture(step, values):
        values_array = np.asarray(values)
        require(np.isfinite(values_array).all(), "independent replay is nonfinite")
        diagnostics.append([step, step * dt, float(np.min(values_array)),
                            float(np.max(values_array)), norm(values_array, h)])
        if step in wanted:
            snapshots[step] = values_array[crop].copy()

    capture(0, previous)
    capture(1, current)
    for step in range(2, steps + 1):
        if scalar:
            new = [0.0] * len(x)
            for j in range(1, len(x) - 1):
                new[j] = (coefficient * current[j-1]
                          + (2 - 2 * coefficient) * current[j]
                          + coefficient * current[j+1] - previous[j])
        else:
            new = np.zeros_like(current)
            new[1:-1] = 2 * current[1:-1] - previous[1:-1] + coefficient * np.diff(current, n=2)
        finish_endpoints(new, current)
        previous, current = current, new
        capture(step, current)
    return {
        "snapshots": snapshots, "diagnostics": np.asarray(diagnostics),
        "h": h, "dt": dt, "steps": steps, "crop_start": crop_start,
        "full_nodes": len(x), "snapshot_steps": sorted(wanted),
        "x": x[crop], "beta": beta, "q": None if not mur else
        (beta_courant - 1) / (beta_courant + 1),
    }


def validate_raw(out, run_id, n_intervals, boundary, large, result):
    raw = out / "raw" / run_id
    data = table(raw / "snapshots.csv", SNAPSHOT_FIELDS)
    diagnostic = table(raw / "diagnostics.csv", DIAGNOSTIC_FIELDS)
    metadata = json.loads((raw / "run.json").read_text(encoding="utf-8"))
    steps = result["steps"]
    tolerance = 128 * EPS * (steps + 1)
    require(len(data) == (n_intervals + 1) * len(result["snapshot_steps"]),
            run_id + ": snapshot row count")
    require(len(diagnostic) == steps + 1, run_id + ": diagnostic row count")
    discrepancy = 0.0
    stored = {}
    block_size = n_intervals + 1
    for k, step in enumerate(result["snapshot_steps"]):
        block = data[k * block_size:(k + 1) * block_size]
        require(all(integer(row["step"], "snapshot step") == step for row in block),
                run_id + ": snapshot step order")
        require([integer(row["index"], "snapshot index") for row in block] ==
                list(range(block_size)), run_id + ": snapshot index order")
        close([float(row["time"]) for row in block],
              np.full(block_size, step * result["dt"]), 8 * EPS, run_id + ": time")
        close([float(row["x"]) for row in block], result["x"], 8 * EPS, run_id + ": x")
        saved_u = np.asarray([float(row["u"]) for row in block])
        stored[step] = saved_u
        discrepancy = max(discrepancy, close(saved_u, result["snapshots"][step],
                                             tolerance, run_id + ": replay"))
    require([integer(row["step"], "diagnostic step") for row in diagnostic] ==
            list(range(steps + 1)), run_id + ": full time history")
    saved_diagnostics = np.asarray([[float(row[name]) for name in DIAGNOSTIC_FIELDS]
                                    for row in diagnostic])
    close(saved_diagnostics[:, 1], result["diagnostics"][:, 1],
          8 * EPS, run_id + ": diagnostic fixed dt")
    discrepancy = max(discrepancy, close(saved_diagnostics[:, 2:],
                                         result["diagnostics"][:, 2:], tolerance,
                                         run_id + ": diagnostics"))
    expected_metadata = {
        "run_id": run_id, "N": n_intervals, "kind": "large" if large else "short",
        "boundary": "dirichlet" if large else boundary["kind"],
        "beta": result["beta"], "c": 1.0, "h": result["h"], "dt": result["dt"],
        "courant": 0.8, "boundary_courant": None if result["beta"] is None else
        result["beta"] * 0.8, "q": result["q"],
        "domain": [-2, 3] if large else [0, 1], "crop": [0, 1],
        "crop_start": result["crop_start"], "full_nodes": result["full_nodes"],
        "final_time_requested": 1.2, "final_step": steps,
        "snapshot_steps": result["snapshot_steps"],
        "diagnostics_scope": "full-domain", "snapshot_scope": "crop",
    }
    for key, value in expected_metadata.items():
        require(key in metadata, run_id + ": missing metadata " + key)
        if isinstance(value, float):
            require(isinstance(metadata[key], (float, int)), run_id + ": numeric " + key)
            close(metadata[key], value, 8 * EPS, run_id + ": metadata " + key)
        else:
            require(metadata[key] == value, run_id + ": metadata " + key)
    return stored, {
        "run_id": run_id, "N": n_intervals, "step_count": steps,
        "snapshots_checked": len(result["snapshot_steps"]),
        "diagnostic_rows_checked": len(diagnostic),
        "max_absolute_replay_discrepancy": discrepancy,
        "roundoff_agreement_tolerance": tolerance,
        "passed": True,
    }


def validate_metrics(config, out, stored, simulations):
    rows = table(out / "metrics.csv", METRIC_FIELDS)
    expected_count = (len(config["boundaries"])
                      * len(config["discretization"]["interval_counts"])
                      * len(config["discretization"]["snapshot_times"]))
    require(len(rows) == expected_count, "metrics row count/all combinations")
    indexed = {}
    for row in rows:
        key = (row["run_id"], integer(row["step"], "metric step"))
        require(key not in indexed, "duplicate metric row")
        indexed[key] = row
    reports = []
    discrepancy = 0.0
    error_series = {}
    m = config["model"]
    for n_intervals in config["discretization"]["interval_counts"]:
        big_id = f"large-N{n_intervals}"
        large_values = stored[big_id]
        for boundary in config["boundaries"]:
            run_id = f"{boundary['name']}-N{n_intervals}"
            result = simulations[run_id]
            h, dt = result["h"], result["dt"]
            tolerance = 256 * EPS * (result["steps"] + 1)
            beta = boundary["speed_ratio"]
            target = continuous_reflection(beta)
            for step in result["snapshot_steps"]:
                key = (run_id, step)
                require(key in indexed, "missing metric " + str(key))
                row = indexed[key]
                require(integer(row["N"], "metric N") == n_intervals, "metric N mismatch")
                require(row["boundary"] == boundary["kind"], "metric boundary mismatch")
                if beta is None:
                    require(row["beta"] == "", "Dirichlet beta must be blank")
                else:
                    close(float(row["beta"]), beta, 8 * EPS, "metric beta")
                for field, value in [("h", h), ("dt", dt), ("time", step * dt),
                                     ("reflection_target", target)]:
                    close(float(row[field]), value, 8 * EPS, "metric " + field)
                exact = pulse(result["x"] - m["c"] * step * dt,
                              m["pulse"]["center"], m["pulse"]["half_width"],
                              m["pulse"]["power"])
                reflected = pulse(2 - result["x"] - m["c"] * step * dt,
                                  m["pulse"]["center"], m["pulse"]["half_width"],
                                  m["pulse"]["power"])
                current = stored[run_id][step]
                reference = large_values[step]
                errors = {
                    "continuous": current - exact,
                    "short_large": current - reference,
                    "large_continuous": reference - exact,
                    "boundary_exact": current - exact - target * reflected,
                }
                metrics = {}
                for label, error in errors.items():
                    metrics[f"l2_{label}_error"] = norm(error, h)
                    metrics[f"linf_{label}_error"] = float(np.max(np.abs(error)))
                for field, value in metrics.items():
                    discrepancy = max(discrepancy, close(float(row[field]), value,
                                                          tolerance, field))
                if step == result["steps"]:
                    difference = current - reference
                    denominator = inner(reflected, reflected, h)
                    require(denominator > 0, "final reflected template is empty")
                    projection = inner(difference, reflected, h) / denominator
                    residual = norm(difference - projection * reflected, h)
                    template_norm = math.sqrt(denominator)
                    final_fields = {
                        "reflected_projection": projection,
                        "reflected_fit_l2": residual,
                        "reflected_template_l2": template_norm,
                    }
                    for field, value in final_fields.items():
                        discrepancy = max(discrepancy, close(float(row[field]), value,
                                                              tolerance, field))
                    reports.append({
                        "run_id": run_id, "N": n_intervals,
                        "signed_reflection_target": target,
                        "signed_projection": projection,
                        "fit_residual_l2": residual,
                        "template_l2": template_norm,
                        "boundary_exact_l2_error": metrics["l2_boundary_exact_error"],
                        "interpretation": "reported observation; no accuracy/order threshold",
                    })
                else:
                    require(all(row[field] == "" for field in
                                ["reflected_projection", "reflected_fit_l2",
                                 "reflected_template_l2"]),
                            "reflection projection must only be evaluated at final time")
                series_key = (boundary["name"], round(step * dt, 12))
                error_series.setdefault(series_key, []).append(
                    (n_intervals, metrics["l2_boundary_exact_error"],
                     metrics["l2_short_large_error"], metrics["l2_large_continuous_error"]))
    orders = []
    for (boundary, time), series in sorted(error_series.items()):
        series.sort()
        for coarse, fine in zip(series[:-1], series[1:]):
            item = {"boundary": boundary, "time": time, "N_coarse": coarse[0],
                    "N_fine": fine[0], "interpretation": "descriptive; not a hard gate"}
            for index, label in [(1, "boundary_exact"), (2, "short_large"),
                                 (3, "large_continuous")]:
                item[label + "_order"] = (
                    math.log(coarse[index] / fine[index]) / math.log(fine[0] / coarse[0])
                    if coarse[index] > 0 and fine[index] > 0 else None)
            orders.append(item)
    return {"rows_checked": len(rows),
            "max_absolute_metric_discrepancy": discrepancy,
            "final_reflection_observations": reports,
            "observed_orders": orders, "passed": True}


def validate_provenance(config_path, out, run_ids):
    provenance = json.loads((out / "run.json").read_text(encoding="utf-8"))
    require(provenance["status"] == "completed", "production run not completed")
    require(provenance["schema_version"] == 1, "run schema version")
    require(provenance["article_id"] == "tutorial-006", "run article id")
    require(provenance["config_sha256"] == sha256(config_path), "config SHA256 mismatch")
    require(len(provenance["run_ids"]) == len(run_ids) and
            set(provenance["run_ids"]) == set(run_ids), "production run combination list")
    experiment_root = config_path.resolve().parent.parent
    require(set(provenance["source_files"]) == {"src/wave.py", "run_experiments.py"},
            "unexpected/missing production source hash entries")
    for relative, digest in provenance["source_files"].items():
        require(sha256(experiment_root / relative) == digest, "source hash mismatch: " + relative)
    required_artifacts = {"metrics.csv"}
    for run_id in run_ids:
        for filename in ["snapshots.csv", "diagnostics.csv", "run.json"]:
            required_artifacts.add(f"raw/{run_id}/{filename}")
    require(set(provenance["artifact_sha256"]) == required_artifacts,
            "artifact hash coverage mismatch")
    for relative, digest in provenance["artifact_sha256"].items():
        require(sha256(out / relative) == digest, "artifact SHA256 mismatch: " + relative)
    directories = {path.name for path in (out / "raw").iterdir() if path.is_dir()}
    require(directories == set(run_ids), "unrecorded/missing raw run directory")
    return {"artifacts_hashed": len(required_artifacts),
            "config_sha256": provenance["config_sha256"],
            "source_files": provenance["source_files"],
            "production_source_commit": provenance.get("source_commit"),
            "passed": True}


def validate(config_path, out):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    check_design(config)
    algebra = mathematical_tests()
    simulations, stored, run_reports = {}, {}, []
    grids = config["discretization"]["interval_counts"]
    all_ids = [f"large-N{n}" for n in grids]
    all_ids += [f"{b['name']}-N{n}" for n in grids for b in config["boundaries"]]
    provenance = validate_provenance(config_path, out, all_ids)
    scalar_reports = []
    for n_intervals in grids:
        cases = [(f"large-N{n_intervals}", None, True)]
        cases += [(f"{b['name']}-N{n_intervals}", b, False) for b in config["boundaries"]]
        for run_id, boundary, large in cases:
            result = replay(config, n_intervals, boundary, large=large)
            saved, report = validate_raw(out, run_id, n_intervals, boundary, large, result)
            simulations[run_id], stored[run_id] = result, saved
            run_reports.append(report)
            if n_intervals == min(grids):
                baseline = replay(config, n_intervals, boundary, large=large, scalar=True)
                tolerance = report["roundoff_agreement_tolerance"]
                delta = close(baseline["diagnostics"], result["diagnostics"], tolerance,
                              run_id + ": scalar diagnostic replay")
                for step in result["snapshot_steps"]:
                    delta = max(delta, close(baseline["snapshots"][step],
                                             result["snapshots"][step], tolerance,
                                             run_id + ": scalar snapshots"))
                scalar_reports.append({"run_id": run_id,
                                       "max_absolute_discrepancy": delta,
                                       "passed": True})
    metrics = validate_metrics(config, out, stored, simulations)
    return {
        "schema_version": 1, "article_id": "tutorial-006", "status": "passed",
        "validator_sha256": sha256(__file__), "config_sha256": sha256(config_path),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "platform": platform.platform()},
        "independence": "No production modules imported; all 20 cases independently replayed; five N=200 cases also replayed with plain lists.",
        "tolerance_policy": "128*machine_epsilon*(number_of_steps+1) for state/diagnostic roundoff agreement; twice that for recomputed metrics. These are not physical accuracy thresholds.",
        "mathematical_tests": algebra, "provenance": provenance,
        "full_replays": run_reports, "scalar_replays": scalar_reports,
        "metrics": metrics,
        "limits": [
            "Agreement with an independent implementation tests this frozen discrete experiment, not arbitrary PDEs.",
            "The enlarged-domain run uses the same h and dt and is not an exact PDE solution.",
            "The one-reflection continuous reference is only used before a reflected pulse reaches the left boundary.",
            "Observed convergence orders and signed projections are reported without post-hoc pass thresholds.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Existing production output directory; writes checks.json here.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(args.config, args.out)
        code = 0
    except Exception as error:
        report = {"schema_version": 1, "article_id": "tutorial-006", "status": "failed",
                  "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(),
                  "validator_sha256": sha256(__file__)}
        code = 1
    (args.out / "checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "checks": str(args.out / "checks.json")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
