"""ET005-03: Godunov vs uncorrected Roe on the same transonic rarefaction.

All reported values come from actual single-face conservative time updates.
The exact entropy solution is used only for error measurement, never for updating U.
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
from exact_rarefaction import entropy_rarefaction_cell_means

SNAPSHOT_COLS = ("method", "time", "i", "x_left", "x_right", "x_center", "U", "U_exact")
METRIC_COLS = (
    "experiment_id", "case_id", "method", "boundary", "N", "dx", "cfl_requested",
    "time", "dt_last", "steps", "initial_mass", "mass", "exact_mass",
    "net_boundary_flux_integral", "mass_defect", "min", "max", "total_variation",
    "TV_boundary", "L1_cellavg", "reference_kind", "energy", "entropy_boundary_integral",
    "entropy_balance", "stationary_max_difference", "status")
BUDGET_COLS = (
    "method", "step", "time_start", "time_end", "dt", "cfl_actual", "F_left", "F_right",
    "mass_before", "mass_after", "flux_budget_increment", "flux_budget_cumulative",
    "step_mass_defect", "mass_defect", "entropy_before", "entropy_after",
    "entropy_balance", "min", "max", "total_variation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, names, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def registered_config(config, design):
    c = json.loads(Path(config).read_text(encoding="utf-8"))
    d = json.loads(Path(design).read_text(encoding="utf-8"))
    fixed = {
        "contract_version": "v0.2-frozen-2026-10-10",
        "experiment_id": "ET005-03", "case_id": "rarefaction_m1_1",
        "methods": ["Godunov", "Roe_no_entropy_fix"],
        "boundary": "riemann", "domain": [-1.0, 1.0], "N": 200,
        "u_L": -1.0, "u_R": 1.0, "requested_cfl": 0.8,
        "a_bound": 1.0, "times": [0.0, 0.1, 0.25, 0.5],
        "reference_kind": "exact_rarefaction_cellavg"
    }
    if c != fixed:
        raise ValueError("ET005-03 differs from pre-registered v0.2 problem")
    if (d["contract_version"] != c["contract_version"] or
            d["riemann_cases"]["domain"] != c["domain"] or
            d["riemann_cases"]["base_N"] != c["N"] or
            d["riemann_cases"]["rarefaction_m1_1"] != [-1.0, 1.0] or
            d["riemann_cases"]["final_time"] != 0.5 or
            d["base_requested_cfl"] != 0.8 or d["wave_speed_bound"] != 1.0):
        raise ValueError("shared registered design disagrees with ET005-03")
    return c


def energy(u, dx):
    return dx * 0.5 * float(np.dot(u, u))


def run(config, design, out, source_sha):
    cfg = registered_config(config, design)
    start = datetime.now(timezone.utc).isoformat()
    n = cfg["N"]
    dx = 2.0 / n
    dt_cap = dx * cfg["requested_cfl"] / cfg["a_bound"]
    initial = riemann_initial_means(n, cfg["u_L"], cfg["u_R"])
    m0 = dx * float(np.sum(initial))
    e0 = energy(initial, dx)
    if abs(m0) > 1e-12 or abs(e0 - 1.0) > 1e-12:
        raise ArithmeticError("bad integral initialization")
    snapshots, metrics, ledger = [], [], []
    for method in cfg["methods"]:
        u = initial.copy()
        t, count, net_flux, dt_last = 0.0, 0, 0.0, None
        for target in cfg["times"]:
            while target - t > 2e-14:
                t0 = t
                dt = min(dt_cap, target - t0)
                if not (0.0 < dt <= dt_cap + 2e-15 and dt / dx <= 1.0):
                    raise ValueError("invalid positive time step or CFL")
                before = dx * float(np.sum(u))
                ebefore = energy(u, dx)
                updated, faces = euler_step(u, dx, dt, "riemann", method,
                                             left=cfg["u_L"], right=cfg["u_R"])
                after = dx * float(np.sum(updated))
                eafter = energy(updated, dx)
                left_flux, right_flux = float(faces[0]), float(faces[-1])
                increment = dt * (left_flux - right_flux)
                net_flux += increment
                count += 1
                t = t0 + dt
                dt_last = dt
                step_defect = (after - before) - increment
                defect = (after - m0) - net_flux
                # The prescribed exterior traces are -1 and +1, so
                # q(right)-q(left) = (1**3 - (-1)**3)/3 = 2/3.
                entropy_balance = eafter - e0 + (2.0 / 3.0) * t
                ledger.append({
                    "method": method, "step": count, "time_start": t0,
                    "time_end": t, "dt": dt, "cfl_actual": dt / dx,
                    "F_left": left_flux, "F_right": right_flux,
                    "mass_before": before, "mass_after": after,
                    "flux_budget_increment": increment,
                    "flux_budget_cumulative": net_flux,
                    "step_mass_defect": step_defect, "mass_defect": defect,
                    "entropy_before": ebefore, "entropy_after": eafter,
                    "entropy_balance": entropy_balance,
                    "min": float(updated.min()), "max": float(updated.max()),
                    "total_variation": total_variation(updated, "riemann", -1.0, 1.0)
                })
                if abs(step_defect) > 1e-11 or abs(defect) > 1e-11:
                    raise ArithmeticError("nonperiodic conservative face budget failed")
                u = updated
            if abs(t - target) > 2e-14:
                raise ArithmeticError("missing exact snapshot time")
            t = target  # eliminate rounding-scale timestamp drift only
            reference = entropy_rarefaction_cell_means(n, target)
            mass = dx * float(np.sum(u))
            en = energy(u, dx)
            for i in range(n):
                xl, xr = -1.0 + i * dx, -1.0 + (i + 1) * dx
                snapshots.append({
                    "method": method, "time": target, "i": i,
                    "x_left": xl, "x_right": xr, "x_center": 0.5 * (xl + xr),
                    "U": float(u[i]), "U_exact": float(reference[i])
                })
            metrics.append({
                "experiment_id": "ET005-03", "case_id": cfg["case_id"],
                "method": method, "boundary": "riemann",
                "N": n, "dx": dx, "cfl_requested": cfg["requested_cfl"],
                "time": target, "dt_last": dt_last, "steps": count,
                "initial_mass": m0, "mass": mass, "exact_mass": 0.0,
                "net_boundary_flux_integral": net_flux,
                "mass_defect": (mass - m0) - net_flux,
                "min": float(u.min()), "max": float(u.max()),
                "total_variation": total_variation(u, "riemann", -1.0, 1.0),
                "TV_boundary": "riemann_fixed_exterior",
                "L1_cellavg": dx * float(np.abs(u - reference).sum()),
                "reference_kind": cfg["reference_kind"],
                "energy": en, "entropy_boundary_integral": (2.0 / 3.0) * target,
                "entropy_balance": en - e0 + (2.0 / 3.0) * target,
                "stationary_max_difference": (float(np.max(np.abs(u - initial)))
                                               if method == "Roe_no_entropy_fix" else None),
                "status": "computed_unvalidated"
            })
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    data = {
        "snapshots.csv": (SNAPSHOT_COLS, snapshots),
        "metrics.csv": (METRIC_COLS, metrics),
        "step_budget.csv": (BUDGET_COLS, ledger)
    }
    for name, (cols, rows) in data.items():
        write_csv(output / name, cols, rows)
    record = {
        "experiment_id": "ET005-03", "status": "computed_unvalidated",
        "source_git_sha": source_sha, "run_started_utc": start,
        "python": platform.python_version(), "numpy": np.__version__,
        "config_path": str(config), "config_sha256": sha256(config),
        "design_sha256": sha256(design),
        "methods": cfg["methods"], "requested_times": cfg["times"],
        "output_counts": {key: len(rows) for key, (_, rows) in data.items()},
        "output_sha256": {key: sha256(output / key) for key in data},
        "steps_per_method": {method: next(int(x["steps"]) for x in metrics
                                          if x["method"] == method and x["time"] == 0.5)
                             for method in cfg["methods"]},
        "entropy_jumps_theoretical": {
            "uncorrected_Roe_transonic_stationary": 2.0 / 3.0,
            "physical_boundary_entropy_flux_difference": 2.0 / 3.0
        }
    }
    (output / "run.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps({
        "output": str(output), "total_step_rows": len(ledger),
        "final": [{"method": x["method"], "mass": x["mass"],
                   "L1_cellavg": x["L1_cellavg"], "entropy_balance": x["entropy_balance"]}
                  for x in metrics if x["time"] == 0.5],
        "output_sha256": record["output_sha256"]
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    run(args.config, args.design, args.out, args.source_sha)
