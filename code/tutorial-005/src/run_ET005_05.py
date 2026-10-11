"""ET005-05: frozen Godunov versus global LF(alpha=1) CFL experiment.

Reuse the untouched shared-face burgers_fvm.euler_step; LF_alpha1 has a
constant globally prescribed alpha, not a state-dependent wave speed.
These are actual Euler time steps, not interpolated/filtered snapshots.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from burgers_fvm import euler_step, riemann_initial_means, total_variation
from exact_references import exact_shock_cell_means
from run_ET005_02 import unique_half_crossing

METHODS = ("Godunov", "LF_alpha1")
CFLS = (0.4, 0.7, 0.9)
TIMES = (0.0, 0.1, 0.25, 0.5)
N = 200
DX = 2.0 / N
METRIC_COLS = (
    "experiment_id", "case_id", "method", "boundary", "N", "dx",
    "LF_alpha", "cfl_requested", "time", "steps", "dt_last",
    "initial_mass", "mass", "exact_mass", "net_boundary_flux_integral",
    "mass_defect", "min", "max", "total_variation", "TV_boundary",
    "L1_cellavg", "reference_kind", "shock_speed_exact", "shock_x_exact",
    "shock_x_half", "shock_x_half_error", "shock_x_mass",
    "transition_low", "transition_high", "transition_cells",
    "transition_segments", "transition_width",
    "quadratic_content", "status"
)
SNAP_COLS = ("method", "cfl_requested", "N", "time", "i", "x_left",
             "x_right", "x_center", "U", "U_exact")
LEDGER_COLS = (
    "method", "cfl_requested", "step", "time_start", "time_end",
    "dt", "cfl_actual", "F_left", "F_right", "mass_before",
    "mass_after", "flux_budget_increment", "flux_budget_cumulative",
    "step_mass_defect", "mass_defect", "min", "max", "total_variation"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_csv(path, cols, items):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)


def registered_config(config_path, design_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    common = json.loads(Path(design_path).read_text(encoding="utf-8"))
    expected = {
        "contract_version": "v0.2-frozen-2026-10-10",
        "experiment_id": "ET005-05",
        "case_id": "shock_1_0",
        "methods": list(METHODS),
        "boundary": "riemann", "domain": [-1.0, 1.0],
        "N": N, "u_L": 1.0, "u_R": 0.0, "LF_alpha": 1.0,
        "requested_cfl": list(CFLS), "a_bound": 1.0,
        "times": list(TIMES), "snapshot_time": 0.5,
        "reference_kind": "exact_shock_cellavg",
        "transition_level_low": 0.1, "transition_level_high": 0.9
    }
    if config != expected:
        raise ValueError("ET005-05 config does not match pre-registered v0.2 matrix")
    if (common["contract_version"] != expected["contract_version"]
            or common["equation"] != "u_t+(u^2/2)_x=0"
            or common["riemann_cases"]["domain"] != expected["domain"]
            or common["riemann_cases"]["base_N"] != N
            or common["riemann_cases"]["final_time"] != 0.5
            or common["riemann_cases"]["shock_1_0"] != [1.0, 0.0]
            or common["ET005-05_cfl"] != list(CFLS)
            or common["ET005-05_methods"] != list(METHODS)
            or common["wave_speed_bound"] != 1.0):
        raise ValueError("ET005-05 disagrees with frozen shared scientific design")
    return config


def transition_region(u, dx=DX, low=0.1, high=0.9):
    """Count all strict-level cells; record disconnected runs without merging."""
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all() or not (0.0 < low < high < 1.0):
        raise ValueError("invalid transition width input")
    mask = (arr > low) & (arr < high)
    count = int(np.count_nonzero(mask))
    segments = int(bool(mask[0])) + int(np.count_nonzero(mask[1:] & (~mask[:-1])))
    return count, segments, count*dx


def quadratic_content(u, dx=DX):
    return float(0.5*dx*np.dot(u, u))


def run(config_path, design_path, out, source_sha):
    config = registered_config(config_path, design_path)
    started = datetime.now(timezone.utc).isoformat()
    n, dx = config["N"], 2.0/config["N"]
    exact = exact_shock_cell_means(n, config["snapshot_time"])
    initial = riemann_initial_means(n, 1.0, 0.0)
    m0 = dx*float(np.sum(initial))
    if abs(m0-1.0) > 1e-12:
        raise ArithmeticError("unexpected initial exact Riemann mass")
    metrics, snaps, steps, schedules = [], [], [], {}
    for method in METHODS:
        for requested_cfl in CFLS:
            u = initial.copy()
            time_now, count, dt_last, budget = 0.0, 0, None, 0.0
            checkpoint_steps = {}
            dt_history = []
            dt_cap = requested_cfl*dx/config["a_bound"]
            for target in TIMES:
                while target-time_now > 2e-14:
                    t0 = time_now
                    dt = min(dt_cap, target-t0)
                    if not (dt > 0.0 and dt <= dt_cap + 2e-15 and dt/dx <= 1.0):
                        raise ArithmeticError("invalid CFL step or checkpoint scheduler")
                    before_mass = dx*float(np.sum(u))
                    # Core implementation is the same one verified by ET005-01..04.
                    new, faces = euler_step(u, dx, dt, "riemann",
                                           method=method, left=1.0, right=0.0)
                    after_mass = dx*float(np.sum(new))
                    fleft, fright = float(faces[0]), float(faces[-1])
                    inc = dt*(fleft-fright)
                    budget += inc
                    defect_step = (after_mass-before_mass)-inc
                    defect_total = (after_mass-m0)-budget
                    if abs(defect_step)>1e-11 or abs(defect_total)>1e-11:
                        raise ArithmeticError("boundary mass budget failed")
                    count += 1
                    time_now = t0+dt
                    dt_last = dt
                    dt_history.append(dt)
                    steps.append({
                        "method": method, "cfl_requested": requested_cfl,
                        "step": count, "time_start": t0, "time_end": time_now,
                        "dt": dt, "cfl_actual": dt/dx,
                        "F_left": fleft, "F_right": fright,
                        "mass_before": before_mass, "mass_after": after_mass,
                        "flux_budget_increment": inc,
                        "flux_budget_cumulative": budget,
                        "step_mass_defect": defect_step,
                        "mass_defect": defect_total,
                        "min": float(new.min()), "max": float(new.max()),
                        "total_variation": total_variation(new,"riemann",1.,0.)
                    })
                    u = new
                if abs(time_now-target)>2e-14:
                    raise ArithmeticError("missed actual registered checkpoint")
                time_now = target  # rounding-only timestamp correction, no U interpolation
                checkpoint_steps[str(target)] = count
            mass = dx*float(np.sum(u))
            if abs(mass-1.25)>1e-11:
                raise ArithmeticError("shock final mass does not match physical inflow")
            l1 = dx*float(np.abs(u-exact).sum())
            xhalf = unique_half_crossing(u,dx)
            cells, segments, width = transition_region(
                u,dx,config["transition_level_low"],config["transition_level_high"])
            for i in range(n):
                xl, xr = -1.0+i*dx, -1.0+(i+1)*dx
                snaps.append({
                    "method": method, "cfl_requested": requested_cfl,
                    "N": n, "time": 0.5, "i": i, "x_left": xl,
                    "x_right": xr, "x_center": 0.5*(xl+xr),
                    "U": float(u[i]), "U_exact": float(exact[i])
                })
            metrics.append({
                "experiment_id": "ET005-05", "case_id": "shock_1_0",
                "method": method, "boundary": "riemann",
                "N": n, "dx": dx, "LF_alpha": 1.0 if method=="LF_alpha1" else None,
                "cfl_requested": requested_cfl, "time": 0.5,
                "steps": count, "dt_last": dt_last,
                "initial_mass": m0, "mass": mass,
                "exact_mass": 1.25, "net_boundary_flux_integral": budget,
                "mass_defect": (mass-m0)-budget,
                "min": float(u.min()), "max": float(u.max()),
                "total_variation": total_variation(u,"riemann",1.,0.),
                "TV_boundary": "riemann_fixed_exterior",
                "L1_cellavg": l1, "reference_kind": "exact_shock_cellavg",
                "shock_speed_exact": 0.5, "shock_x_exact": 0.25,
                "shock_x_half": xhalf, "shock_x_half_error": xhalf-0.25,
                "shock_x_mass": -1.0+mass,
                "transition_low": config["transition_level_low"],
                "transition_high": config["transition_level_high"],
                "transition_cells": cells, "transition_segments": segments,
                "transition_width": width, "quadratic_content": quadratic_content(u,dx),
                "status": "computed_unvalidated"
            })
            schedules[f"{method}:CFL={requested_cfl}"] = {
                "steps": count, "checkpoints_steps": checkpoint_steps,
                "dt_count": len(dt_history)
            }
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    data = {"cfl_flux_metrics.csv": (METRIC_COLS,metrics),
            "snapshots.csv": (SNAP_COLS,snaps),
            "step_budget.csv": (LEDGER_COLS,steps)}
    for name,(columns,items) in data.items():
        dump_csv(output/name,columns,items)
    provenance = {
        "experiment_id": "ET005-05", "status": "computed_unvalidated",
        "source_git_sha": source_sha, "run_started_utc": started,
        "python": platform.python_version(), "numpy": np.__version__,
        "config_path": str(config_path),
        "config_sha256": sha(config_path), "design_sha256": sha(design_path),
        "methods": list(METHODS), "cfl_requested": list(CFLS),
        "requested_times": list(TIMES), "alpha_LF": 1.0,
        "step_schedules": schedules,
        "output_counts": {name:len(items) for name,(_,items) in data.items()},
        "output_sha256": {name:sha(output/name) for name in data}
    }
    (output/"run.json").write_text(
        json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "results": [{"method":r["method"],"CFL":r["cfl_requested"],
                     "steps":r["steps"],"L1_cellavg":r["L1_cellavg"],
                     "shock_x_half":r["shock_x_half"],
                     "transition_width":r["transition_width"],
                     "transition_segments":r["transition_segments"],
                     "mass_defect":r["mass_defect"]} for r in metrics],
        "csv_sha256": provenance["output_sha256"]
    },sort_keys=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--design",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    args=p.parse_args()
    run(args.config,args.design,args.out,args.source_sha)
