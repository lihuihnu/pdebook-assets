"""ET005-02: frozen Riemann shock, single-face Godunov FVM, honest flux ledger."""
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


SNAPSHOT_COLS = ("time", "i", "x_left", "x_right", "x_center", "U", "U_exact")
METRIC_COLS = (
    "experiment_id", "case_id", "method", "boundary", "N", "dx", "cfl_requested",
    "time", "dt_last", "steps", "initial_mass", "mass", "exact_mass",
    "net_boundary_flux_integral", "mass_defect", "min", "max", "total_variation",
    "TV_boundary", "L1_cellavg", "reference_kind", "shock_speed_exact",
    "shock_x_exact", "shock_x_half", "shock_x_half_error", "shock_x_mass", "status")
BUDGET_COLS = (
    "step", "time_start", "time_end", "dt", "cfl_actual", "F_left", "F_right",
    "mass_before", "mass_after", "flux_budget_increment", "flux_budget_cumulative",
    "step_mass_defect", "mass_defect", "min", "max", "total_variation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, columns, values):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def registered_config(config_path, design_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    fixed = {
        "contract_version": "v0.2-frozen-2026-10-10",
        "experiment_id": "ET005-02", "case_id": "shock_1_0", "method": "Godunov",
        "boundary": "riemann", "domain": [-1.0, 1.0], "N": 200,
        "u_L": 1.0, "u_R": 0.0, "requested_cfl": 0.8, "a_bound": 1.0,
        "times": [0.0, 0.1, 0.25, 0.5], "reference_kind": "exact_shock_cellavg",
    }
    if cfg != fixed:
        raise ValueError("ET005-02 configuration differs from frozen v0.2 registration")
    if (design["contract_version"] != fixed["contract_version"]
            or design["riemann_cases"]["domain"] != fixed["domain"]
            or design["riemann_cases"]["base_N"] != fixed["N"]
            or design["riemann_cases"]["shock_1_0"] != [1.0, 0.0]
            or design["riemann_cases"]["final_time"] != 0.5
            or design["base_requested_cfl"] != 0.8
            or design["wave_speed_bound"] != 1.0):
        raise ValueError("ET005-02 no longer agrees with frozen shared design")
    return cfg


def unique_half_crossing(u, dx):
    """First and only descending half-level crossing at neighboring cell centers."""
    centers = -1.0 + (np.arange(len(u), dtype=np.float64) + 0.5) * dx
    crossings = np.flatnonzero((u[:-1] >= 0.5) & (u[1:] < 0.5))
    if crossings.size != 1:
        raise ValueError(f"expected unique descending half-level crossing, got {crossings.size}")
    j = int(crossings[0])
    delta = float(u[j] - u[j+1])
    if delta <= 0.0:
        raise ValueError("invalid half-level slope")
    return float(centers[j] + dx * ((float(u[j]) - 0.5) / delta))


def run(config_path, design_path, out, source_sha):
    cfg = registered_config(config_path, design_path)
    started_at = datetime.now(timezone.utc).isoformat()
    n = cfg["N"]
    dx = (cfg["domain"][1] - cfg["domain"][0]) / n
    dt_bound = cfg["requested_cfl"] * dx / cfg["a_bound"]
    u = riemann_initial_means(n, cfg["u_L"], cfg["u_R"])
    m0 = dx * float(np.sum(u))
    if abs(m0 - 1.0) > 1e-13:
        raise ArithmeticError("bad exact initial Riemann cell averages")
    t, steps, dt_last, budget_integral = 0.0, 0, None, 0.0
    snapshots, metrics, budget = [], [], []
    for target in cfg["times"]:
        while target - t > 2e-14:
            before_time = t
            dt = min(dt_bound, target - before_time)
            if not (0 < dt <= dt_bound + 2e-15 and dt / dx <= 1.0):
                raise ValueError("positive CFL bound violated")
            m_before = dx * float(np.sum(u))
            new, faces = euler_step(u, dx, dt, cfg["boundary"], cfg["method"],
                                    left=cfg["u_L"], right=cfg["u_R"])
            m_after = dx * float(np.sum(new))
            flux_left, flux_right = float(faces[0]), float(faces[-1])
            increment = dt * (flux_left - flux_right)
            steps += 1
            t = before_time + dt
            dt_last = dt
            budget_integral += increment
            step_defect = (m_after - m_before) - increment
            mass_defect = (m_after - m0) - budget_integral
            budget.append({
                "step": steps, "time_start": before_time, "time_end": t, "dt": dt,
                "cfl_actual": dt / dx, "F_left": flux_left, "F_right": flux_right,
                "mass_before": m_before, "mass_after": m_after,
                "flux_budget_increment": increment,
                "flux_budget_cumulative": budget_integral,
                "step_mass_defect": step_defect, "mass_defect": mass_defect,
                "min": float(np.min(new)), "max": float(np.max(new)),
                "total_variation": total_variation(new, "riemann", 1., 0.),
            })
            if abs(step_defect) > 1e-11 or abs(mass_defect) > 1e-11:
                raise ArithmeticError("boundary flux ledger violates conservation")
            u = new
        if abs(t - target) > 2e-14:
            raise ArithmeticError("missed exact registered output time")
        t = target  # only remove roundoff; never interpolate or overwrite U
        exact = exact_shock_cell_means(n, target)
        x_theory = 0.5 * target
        x_half = unique_half_crossing(u, dx)
        mass = dx * float(np.sum(u))
        L1 = dx * float(np.abs(u - exact).sum())
        for i in range(n):
            xl = -1.0 + i * dx
            xr = -1.0 + (i + 1) * dx
            snapshots.append({"time": target, "i": i, "x_left": xl,
                              "x_right": xr, "x_center": 0.5*(xl+xr),
                              "U": float(u[i]), "U_exact": float(exact[i])})
        metrics.append({
            "experiment_id": cfg["experiment_id"], "case_id": cfg["case_id"],
            "method": cfg["method"], "boundary": cfg["boundary"], "N": n, "dx": dx,
            "cfl_requested": cfg["requested_cfl"], "time": target,
            "dt_last": dt_last, "steps": steps, "initial_mass": m0,
            "mass": mass, "exact_mass": 1.0 + target/2.0,
            "net_boundary_flux_integral": budget_integral,
            "mass_defect": (mass-m0) - budget_integral,
            "min": float(np.min(u)), "max": float(np.max(u)),
            "total_variation": total_variation(u, "riemann", 1., 0.),
            "TV_boundary": "riemann_fixed_exterior",
            "L1_cellavg": L1, "reference_kind": cfg["reference_kind"],
            "shock_speed_exact": 0.5, "shock_x_exact": x_theory,
            "shock_x_half": x_half, "shock_x_half_error": x_half-x_theory,
            "shock_x_mass": -1.0+mass,
            "status": "computed_unvalidated",
        })
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "snapshots.csv": (SNAPSHOT_COLS, snapshots),
        "metrics.csv": (METRIC_COLS, metrics),
        "step_budget.csv": (BUDGET_COLS, budget),
    }
    for name, (columns, entries) in data.items():
        csv_write(output/name, columns, entries)
    record = {
        "experiment_id": "ET005-02", "status": "computed_unvalidated",
        "run_started_utc": started_at, "source_git_sha": source_sha,
        "python": platform.python_version(), "numpy": np.__version__,
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "design_sha256": sha256(design_path),
        "output_counts": {name: len(entries) for name, (_, entries) in data.items()},
        "output_sha256": {name: sha256(output/name) for name in data},
        "terminal_target": cfg["times"][-1],
        "requested_times": cfg["times"],
        "time_steps_per_checkpoint": {str(row["time"]): row["steps"] for row in metrics},
    }
    (output/"run.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "steps": steps,
                      "final_mass": metrics[-1]["mass"],
                      "final_L1_cellavg": metrics[-1]["L1_cellavg"],
                      "final_x_half": metrics[-1]["shock_x_half"],
                      "max_mass_defect": max(abs(row["mass_defect"]) for row in metrics),
                      "csv_sha256": record["output_sha256"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    run(args.config, args.design, args.out, args.source_sha)


if __name__ == "__main__":
    main()
