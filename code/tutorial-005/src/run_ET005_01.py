"""Frozen ET005-01: actual periodic Godunov evolution and transparent observations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from burgers_fvm import euler_step, periodic_initial_means, total_variation
from exact_references import BREAK_TIME, smooth_reference

SNAPSHOT_COLS = ("time", "i", "x_left", "x_right", "x_center", "U")
METRIC_COLS = (
    "experiment_id", "case_id", "method", "N", "dx", "cfl_requested", "time", "dt_last",
    "steps", "initial_mass", "mass", "net_boundary_flux_integral", "mass_defect", "min",
    "max", "total_variation", "TV_boundary", "L1_cellavg", "reference_kind",
    "max_characteristic_residual", "max_reference_quadrature_disagreement", "periodic_xi_shift_error",
    "shock_theory_x", "shock_proxy_x", "shock_proxy_error", "shock_proxy_ambiguous", "status")
BUDGET_COLS = ("step", "time_start", "time_end", "dt", "cfl_actual", "F_left", "F_right",
               "mass_before", "mass_after", "flux_budget_increment", "flux_budget_cumulative",
               "mass_defect", "min", "max", "total_variation")
REF_COLS = ("time", "i", "x_left", "x_right", "x_center", "U_exact", "U_quad", "abs_discrepancy")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, names, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def validated_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    case = cfg["periodic_case"]
    expected = {"experiment_id": "ET005-01", "case_id": "smooth_periodic_sine",
                "method": "Godunov", "boundary": "periodic", "domain": [0.0, 1.0], "N": 200,
                "initial": "0.5+0.25*sin(2*pi*x)", "times": [0.0, 0.3, 0.6, 0.7, 0.8],
                "reference_times": [0.0, 0.3, 0.6]}
    if case != expected or cfg["contract_version"] != "v0.2-frozen-2026-10-10":
        raise ValueError("ET005-01 parameters depart from frozen contract")
    if cfg["base_requested_cfl"] != 0.8 or cfg["wave_speed_bound"] != 1.0:
        raise ValueError("ET005-01 CFL or speed bound changed")
    return cfg, case


def shock_jump_diagnostic(u, dx, time):
    if time <= BREAK_TIME:
        return (None, None, None, None)
    # A numerical grid proxy, not the exact entropy-shock interface.
    jumps = np.roll(u, -1) - u
    minimum = float(np.min(jumps))
    ties = np.flatnonzero(np.abs(jumps - minimum) <= 1e-12)
    x_theory = (0.5 + 0.5 * time) % 1.0
    if ties.size != 1:
        return (x_theory, None, None, True)
    x_proxy = float(((int(ties[0]) + 1) * dx) % 1.0)
    error = (x_proxy - x_theory + 0.5) % 1.0 - 0.5
    return (x_theory, x_proxy, float(error), False)


def run(config_path, out, source_sha):
    started_at = datetime.now(timezone.utc).isoformat()
    cfg, c = validated_config(config_path)
    n = c["N"]
    dx = 1.0 / n
    dt_base = dx * cfg["base_requested_cfl"] / cfg["wave_speed_bound"]
    u = periodic_initial_means(n)
    initial_mass = dx * float(np.sum(u))
    current_time = 0.0
    count = 0
    dt_last = None
    mass_flux_integral = 0.0
    snapshot_rows, metric_rows, step_rows, reference_rows = [], [], [], []
    for target in c["times"]:
        while target - current_time > 2e-14:
            t0 = current_time
            dt = min(dt_base, target - t0)
            if dt <= 0 or dt / dx > 1 + 5e-14:
                raise ValueError("invalid positive CFL/time step")
            mass_before = dx * float(np.sum(u))
            new, face = euler_step(u, dx, dt, "periodic", "Godunov")
            count += 1
            dt_last = dt
            current_time = t0 + dt
            # Same periodic physical face: flux left and right are identical.
            increment = dt * (float(face[0]) - float(face[-1]))
            mass_flux_integral += increment
            mass_after = dx * float(np.sum(new))
            defect = mass_after - initial_mass - mass_flux_integral
            step_rows.append({
                "step": count, "time_start": t0, "time_end": current_time, "dt": dt,
                "cfl_actual": dt / dx, "F_left": float(face[0]), "F_right": float(face[-1]),
                "mass_before": mass_before, "mass_after": mass_after,
                "flux_budget_increment": increment, "flux_budget_cumulative": mass_flux_integral,
                "mass_defect": defect, "min": float(np.min(new)), "max": float(np.max(new)),
                "total_variation": total_variation(new, "periodic")})
            u = new
            if abs(defect) > 1e-11:
                raise ArithmeticError("mass defect exceeded registration tolerance")
        if abs(target - current_time) > 2e-14:
            raise ArithmeticError("did not reach requested output time")
        current_time = target  # Remove only rounding-scale arithmetic drift.
        for i, value in enumerate(u):
            snapshot_rows.append({"time": target, "i": i, "x_left": i * dx,
                                  "x_right": (i + 1) * dx, "x_center": (i + 0.5) * dx,
                                  "U": float(value)})
        smooth = target < BREAK_TIME
        ref = smooth_reference(n, target) if smooth else None
        if ref is not None:
            if ref["max_root_residual"] > 2e-13 or ref["max_quad_disagreement"] > 2e-10:
                raise ArithmeticError("smooth reference independent QA failed")
            for i in range(n):
                reference_rows.append({"time": target, "i": i, "x_left": i * dx,
                                       "x_right": (i + 1) * dx, "x_center": (i + 0.5) * dx,
                                       "U_exact": float(ref["exact"][i]),
                                       "U_quad": float(ref["quad"][i]),
                                       "abs_discrepancy": float(abs(ref["exact"][i] - ref["quad"][i]))})
        xs, xp, xe, ambiguous = shock_jump_diagnostic(u, dx, target)
        mass = dx * float(np.sum(u))
        metric_rows.append({
            "experiment_id": c["experiment_id"], "case_id": c["case_id"], "method": c["method"],
            "N": n, "dx": dx, "cfl_requested": cfg["base_requested_cfl"],
            "time": target, "dt_last": dt_last, "steps": count,
            "initial_mass": initial_mass, "mass": mass,
            "net_boundary_flux_integral": mass_flux_integral,
            "mass_defect": mass - initial_mass - mass_flux_integral,
            "min": float(np.min(u)), "max": float(np.max(u)),
            "total_variation": total_variation(u, "periodic"), "TV_boundary": "periodic",
            "L1_cellavg": dx * float(np.abs(u - ref["exact"]).sum()) if smooth else None,
            "reference_kind": "smooth_exact_cellavg" if smooth else "not_available_post_break",
            "max_characteristic_residual": ref["max_root_residual"] if smooth else None,
            "max_reference_quadrature_disagreement": ref["max_quad_disagreement"] if smooth else None,
            "periodic_xi_shift_error": ref["periodic_xi_shift_error"] if smooth else None,
            "shock_theory_x": xs, "shock_proxy_x": xp, "shock_proxy_error": xe,
            "shock_proxy_ambiguous": ambiguous if not smooth else None, "status": "computed_unvalidated"})
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    outputs = {"snapshots.csv": (SNAPSHOT_COLS, snapshot_rows),
               "metrics.csv": (METRIC_COLS, metric_rows),
               "step_budget.csv": (BUDGET_COLS, step_rows),
               "reference_cellavg.csv": (REF_COLS, reference_rows)}
    for name, (columns, rows) in outputs.items():
        write_csv(output / name, columns, rows)
    meta = {
        "experiment_id": "ET005-01", "status": "computed_unvalidated",
        "source_git_sha": source_sha, "run_started_utc": started_at,
        "python": platform.python_version(), "numpy": np.__version__,
        "config_path": str(config_path), "config_sha256": sha_file(config_path),
        "output_sha256": {name: sha_file(output / name) for name in outputs},
        "output_counts": {name: len(rows) for name, (_, rows) in outputs.items()},
        "break_time_analytic": BREAK_TIME,
        "time_steps_per_checkpoint": {str(row["time"]): row["steps"] for row in metric_rows},
    }
    (output / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "steps": count,
                      "mass_defect_max": max(abs(r["mass_defect"]) for r in metric_rows),
                      "break_time": BREAK_TIME, "csv_sha256": meta["output_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--source-sha", required=True)
    args = p.parse_args()
    run(args.config, args.out, args.source_sha)
