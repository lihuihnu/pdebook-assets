"""ET005-04: pre-registered shock/rarefaction four-grid Godunov refinement.

Run the unchanged ET005-01--03 single-face FVM core. Only final-time cell
averages are saved as snapshots; the entire accepted step ledger is retained.
No PDE calls from figure rendering and no post-hoc grid/parameter selection.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from burgers_fvm import euler_step, riemann_initial_means, total_variation
from exact_references import exact_shock_cell_means
from exact_rarefaction import entropy_rarefaction_cell_means
from run_ET005_02 import unique_half_crossing

GRID_COLS = (
    "experiment_id", "case_id", "method", "boundary", "N", "dx",
    "cfl_requested", "time", "steps", "dt_last", "initial_mass", "mass",
    "exact_mass", "net_boundary_flux_integral", "mass_defect",
    "minimum", "maximum", "total_variation", "TV_boundary",
    "L1_cellavg", "reference_kind", "previous_N", "p_obs_from_previous",
    "shock_speed_exact", "shock_x_exact", "shock_x_half",
    "shock_x_half_error", "status"
)
SNAP_COLS = (
    "case_id", "method", "N", "time", "i", "x_left", "x_right",
    "x_center", "U", "U_exact"
)
LEDGER_COLS = (
    "case_id", "method", "N", "step", "time_start", "time_end",
    "dt", "cfl_actual", "F_left", "F_right", "mass_before",
    "mass_after", "flux_budget_increment", "flux_budget_cumulative",
    "step_mass_defect", "mass_defect", "minimum", "maximum",
    "total_variation"
)
TIMES = [0.0, 0.1, 0.25, 0.5]
NS = [100, 200, 400, 800]
CASES = ["shock_1_0", "rarefaction_m1_1"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def frozen_config(config_path, design_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    expected = {
        "contract_version": "v0.2-frozen-2026-10-10",
        "experiment_id": "ET005-04", "method": "Godunov",
        "boundary": "riemann", "domain": [-1, 1],
        "cases": CASES, "N": NS,
        "requested_cfl": 0.8, "a_bound": 1,
        "times": TIMES, "reference_kind": "exact_entropy_solution_cellavg",
        "output_snapshot_time": 0.5
    }
    if config != expected:
        raise ValueError("ET005-04 differs from frozen pre-registered 2x4 matrix")
    if (design["contract_version"] != expected["contract_version"]
            or design["equation"] != "u_t+(u^2/2)_x=0"
            or design["riemann_cases"]["domain"] != [-1.0, 1.0]
            or design["riemann_cases"]["shock_1_0"] != [1.0, 0.0]
            or design["riemann_cases"]["rarefaction_m1_1"] != [-1.0, 1.0]
            or design["riemann_cases"]["base_N"] != 200
            or design["riemann_cases"]["final_time"] != 0.5
            or design["ET005-04_N"] != NS
            or design["base_requested_cfl"] != 0.8
            or design["wave_speed_bound"] != 1.0):
        raise ValueError("shared v0.2 design differs from registered ET005-04")
    return config


def exact_means(case, n, t):
    if case == "shock_1_0":
        return exact_shock_cell_means(n, t)
    if case == "rarefaction_m1_1":
        return entropy_rarefaction_cell_means(n, t)
    raise ValueError("unknown registered Riemann case")


def one_case(case, n, config, step_rows, snap_rows):
    dx = 2.0/n
    dt_cap = config["requested_cfl"]*dx/config["a_bound"]
    ul, ur = (1.0, 0.0) if case == "shock_1_0" else (-1.0, 1.0)
    u = riemann_initial_means(n, ul, ur)
    m0 = dx*float(np.sum(u))
    start = time.perf_counter()
    count, t, cumulative, dt_last = 0, 0.0, 0.0, None
    for target in config["times"]:
        while target-t > 2e-14:
            before_time = t
            dt = min(dt_cap, target-before_time)
            if not (0.0 < dt <= dt_cap+2e-15 and dt/dx <= 1.0):
                raise ArithmeticError("invalid positive CFL / time step")
            mbefore = dx*float(np.sum(u))
            new, faces = euler_step(u, dx, dt, "riemann", "Godunov",
                                    left=ul, right=ur)
            mafter = dx*float(np.sum(new))
            fleft, fright = float(faces[0]), float(faces[-1])
            increment = dt*(fleft-fright)
            cumulative += increment
            count += 1
            t = before_time+dt
            dt_last = dt
            step_defect = (mafter-mbefore)-increment
            defect = (mafter-m0)-cumulative
            if abs(step_defect) > 1e-11 or abs(defect) > 1e-11:
                raise ArithmeticError("violated shared-face mass ledger")
            step_rows.append({
                "case_id": case, "method": "Godunov", "N": n,
                "step": count, "time_start": before_time, "time_end": t,
                "dt": dt, "cfl_actual": dt/dx,
                "F_left": fleft, "F_right": fright,
                "mass_before": mbefore, "mass_after": mafter,
                "flux_budget_increment": increment,
                "flux_budget_cumulative": cumulative,
                "step_mass_defect": step_defect, "mass_defect": defect,
                "minimum": float(new.min()), "maximum": float(new.max()),
                "total_variation": total_variation(new, "riemann", ul, ur)
            })
            u = new
        if abs(t-target) > 2e-14:
            raise ArithmeticError("missed pre-registered checkpoint")
        t = target  # only arithmetic timestamp drift, not U interpolation
    exact = exact_means(case, n, config["output_snapshot_time"])
    l1 = dx*float(np.abs(u-exact).sum())
    mass = dx*float(np.sum(u))
    exact_mass = 1.25 if case == "shock_1_0" else 0.0
    if abs(mass-exact_mass) > 1e-11:
        raise ArithmeticError("incorrect analytical nonperiodic mass balance")
    shock_half = unique_half_crossing(u, dx) if case == "shock_1_0" else None
    for i in range(n):
        xl, xr = -1.0+i*dx, -1.0+(i+1)*dx
        snap_rows.append({
            "case_id": case, "method": "Godunov", "N": n,
            "time": 0.5, "i": i, "x_left": xl, "x_right": xr,
            "x_center": 0.5*(xl+xr), "U": float(u[i]),
            "U_exact": float(exact[i])
        })
    result = {
        "experiment_id": "ET005-04", "case_id": case,
        "method": "Godunov", "boundary": "riemann", "N": n, "dx": dx,
        "cfl_requested": config["requested_cfl"], "time": 0.5,
        "steps": count, "dt_last": dt_last,
        "initial_mass": m0, "mass": mass, "exact_mass": exact_mass,
        "net_boundary_flux_integral": cumulative,
        "mass_defect": (mass-m0)-cumulative,
        "minimum": float(u.min()), "maximum": float(u.max()),
        "total_variation": total_variation(u, "riemann", ul, ur),
        "TV_boundary": "riemann_fixed_exterior",
        "L1_cellavg": l1, "reference_kind": config["reference_kind"],
        "previous_N": None, "p_obs_from_previous": None,
        "shock_speed_exact": 0.5 if shock_half is not None else None,
        "shock_x_exact": 0.25 if shock_half is not None else None,
        "shock_x_half": shock_half,
        "shock_x_half_error": shock_half-0.25 if shock_half is not None else None,
        "status": "computed_unvalidated"
    }
    elapsed = time.perf_counter()-start
    return result, elapsed


def run(config_path, design_path, out, source_sha):
    cfg = frozen_config(config_path, design_path)
    started = datetime.now(timezone.utc).isoformat()
    grid_rows, snaps, ledger, timings = [], [], [], {}
    for case in CASES:
        previous_n, previous_error = None, None
        for n in NS:
            row, elapsed = one_case(case, n, cfg, ledger, snaps)
            current = row["L1_cellavg"]
            if previous_n is not None:
                row["previous_N"] = previous_n
                if (math.isfinite(previous_error) and math.isfinite(current)
                        and previous_error > 0.0 and current > 0.0):
                    row["p_obs_from_previous"] = math.log(previous_error/current, 2.0)
            grid_rows.append(row)
            timings[f"{case}:N={n}"] = elapsed
            previous_n, previous_error = n, current
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "grid_errors.csv": (GRID_COLS, grid_rows),
        "snapshots.csv": (SNAP_COLS, snaps),
        "step_budget.csv": (LEDGER_COLS, ledger)
    }
    for name, (columns, rows) in data.items():
        write_csv(output/name, columns, rows)
    record = {
        "experiment_id": "ET005-04",
        "status": "computed_unvalidated",
        "source_git_sha": source_sha,
        "run_started_utc": started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "design_sha256": sha256(design_path),
        "archive_cases": CASES,
        "archive_N": NS,
        "checkpoints": TIMES,
        "terminal_time": 0.5,
        "elapsed_wall_seconds_by_case_N": timings,
        "output_counts": {name: len(rows) for name, (_, rows) in data.items()},
        "output_sha256": {name: sha256(output/name) for name in data}
    }
    (output/"run.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n",
                                   encoding="utf-8")
    print(json.dumps({
        "result_dir": str(output),
        "matrix_rows": len(grid_rows),
        "snapshots": len(snaps),
        "ledger_steps": len(ledger),
        "cases": [{"case": r["case_id"], "N": r["N"],
                   "L1": r["L1_cellavg"],
                   "p_obs": r["p_obs_from_previous"],
                   "steps": r["steps"]}
                  for r in grid_rows],
        "csv_sha256": record["output_sha256"]
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    a = parser.parse_args()
    run(a.config, a.design, a.out, a.source_sha)
