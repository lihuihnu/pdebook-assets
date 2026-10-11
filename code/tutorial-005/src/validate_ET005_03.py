"""Independent ET005-03 fail-closed audit; never imports the production PDE solver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

N = 200
DX = 2.0 / N
CFL = 0.8
TIMES = (0.0, 0.1, 0.25, 0.5)
METHODS = ("Godunov", "Roe_no_entropy_fix")
LEFT, RIGHT = -1.0, 1.0
SNAP_COLS = ("method", "time", "i", "x_left", "x_right", "x_center", "U", "U_exact")
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


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path, columns):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == list(columns), "unexpected CSV columns: " + str(path))
        return list(reader)


def near(actual, expected, label, tol=2e-12):
    value = float(actual)
    require(math.isfinite(value) and math.isfinite(float(expected))
            and abs(value - float(expected)) <= tol,
            f"{label}: measured={value}, expected={expected}")
    return value


def scalar_flux(a, b, method):
    """Independent scalar formula; does not import burgers_flux."""
    fa, fb = 0.5 * a * a, 0.5 * b * b
    if method == "Godunov":
        if a > b:
            return max(fa, fb)
        if a >= 0.0:
            return fa
        if b <= 0.0:
            return fb
        return 0.0
    if method == "Roe_no_entropy_fix":
        return 0.5 * (fa + fb) - 0.25 * abs(a + b) * (b - a)
    raise ValueError("unregistered independent method")


def independent_integral_reference(t):
    """Integrate each region's exact continuous entropy solution geometrically."""
    out = np.empty(N, dtype=np.float64)
    def overlap(a, b, low, high):
        return max(0.0, min(b, high) - max(a, low))
    for i in range(N):
        a, b = -1.0 + DX * i, -1.0 + DX * (i + 1)
        if t == 0.0:
            out[i] = (overlap(a, b, -1.0, 0.0) * (-1.0)
                      + overlap(a, b, 0.0, 1.0)) / DX
            continue
        left_len = overlap(a, b, -1.0, -t)
        right_len = overlap(a, b, t, 1.0)
        l, r = max(a, -t), min(b, t)
        fan_integral = 0.5 * (r * r - l * l) / t if r > l else 0.0
        out[i] = (-left_len + fan_integral + right_len) / DX
    return out


def energy(u):
    return 0.5 * DX * float(np.dot(u, u))


def variation(u):
    return (float(np.abs(np.diff(u)).sum()) +
            abs(float(u[0] - LEFT)) + abs(float(RIGHT - u[-1])))


def audit(out_path, config_path, design_path):
    out = Path(out_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    expected_cfg = {
        "contract_version": "v0.2-frozen-2026-10-10",
        "experiment_id": "ET005-03", "case_id": "rarefaction_m1_1",
        "methods": list(METHODS), "boundary": "riemann",
        "domain": [-1.0, 1.0], "N": N, "u_L": LEFT, "u_R": RIGHT,
        "requested_cfl": CFL, "a_bound": 1.0, "times": list(TIMES),
        "reference_kind": "exact_rarefaction_cellavg"
    }
    require(config == expected_cfg, "frozen ET005-03 config changed")
    require(design["contract_version"] == expected_cfg["contract_version"]
            and design["riemann_cases"]["domain"] == [-1.0, 1.0]
            and design["riemann_cases"]["base_N"] == N
            and design["riemann_cases"]["rarefaction_m1_1"] == [-1.0, 1.0]
            and design["riemann_cases"]["final_time"] == 0.5
            and design["base_requested_cfl"] == CFL
            and design["wave_speed_bound"] == 1.0,
            "shared pre-registered design changed")
    require(near(scalar_flux(-1.0, 1.0, "Godunov"), 0.0, "Godunov sonic", 0.0) == 0.0,
            "wrong Godunov sign")
    near(scalar_flux(-1.0, 1.0, "Roe_no_entropy_fix"), 0.5, "uncorrected Roe", 0.0)
    entropy_jump_defect = ((RIGHT**3 - LEFT**3) / 3.0)  # s=0, [eta]=0
    near(entropy_jump_defect, 2.0 / 3.0, "stationary jump entropy violation", 0.0)
    info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    require(info["experiment_id"] == "ET005-03"
            and info["status"] == "computed_unvalidated"
            and isinstance(info["source_git_sha"], str) and info["source_git_sha"],
            "missing raw run provenance/status")
    require(info["config_sha256"] == digest(config_path)
            and info["design_sha256"] == digest(design_path), "changed input SHA256")
    require(info["methods"] == list(METHODS) and info["requested_times"] == list(TIMES),
            "run identity differs from frozen design")
    required = ("snapshots.csv", "metrics.csv", "step_budget.csv")
    require(set(info["output_sha256"]) == set(required), "missing or extra raw results")
    for path in required:
        require(digest(out / path) == info["output_sha256"][path],
                "output SHA256 mismatch: " + path)
    snaps = load_csv(out / "snapshots.csv", SNAP_COLS)
    metrics = load_csv(out / "metrics.csv", METRIC_COLS)
    ledger = load_csv(out / "step_budget.csv", BUDGET_COLS)
    require(len(snaps) == 2 * 4 * N and len(metrics) == 8 and len(ledger) == 128,
            "raw row counts mismatch")
    require(info["output_counts"] == {
        "snapshots.csv": 1600, "metrics.csv": 8, "step_budget.csv": 128},
        "run metadata row counts mismatch")
    require(info["steps_per_method"] == {method: 64 for method in METHODS},
            "methods did not run same time grid")
    expected_metric_keys = [(method, t) for method in METHODS for t in TIMES]
    require([(r["method"], float(r["time"])) for r in metrics] == expected_metric_keys,
            "missing, duplicated or reordered method/time metrics")
    init = np.array([-1.0] * (N // 2) + [1.0] * (N // 2), dtype=np.float64)
    m0 = DX * float(init.sum())
    e0 = energy(init)
    require(abs(m0) < 1e-15 and abs(e0 - 1.0) < 1e-15, "independent initial means incorrect")
    max_replay_state_error = max_budget_error = max_flux_error = 0.0
    results = {}
    dt_sequences = {}
    for imethod, method in enumerate(METHODS):
        u = init.copy()
        count, t, budget_integral, last_dt = 0, 0.0, 0.0, None
        dt_sequence = []
        previous_tv = variation(u)
        method_rows = metrics[imethod*4:(imethod+1)*4]
        for it, target in enumerate(TIMES):
            while target - t > 2e-14:
                row = ledger[imethod*64 + count]
                require(row["method"] == method and int(row["step"]) == count + 1,
                        "ledger method or step index wrong")
                near(row["time_start"], t, "time ledger start", 3e-14)
                dt = min(CFL * DX, target - t)
                near(row["dt"], dt, "time scheduler", 3e-14)
                require(0.0 < dt <= CFL * DX + 3e-14
                        and 0.0 < dt/DX <= 1.0, "invalid CFL")
                near(row["cfl_actual"], dt/DX, "actual CFL", 3e-13)
                faces = [scalar_flux(LEFT, float(u[0]), method)]
                for j in range(1, N):
                    faces.append(scalar_flux(float(u[j-1]), float(u[j]), method))
                faces.append(scalar_flux(float(u[-1]), RIGHT, method))
                max_flux_error = max(max_flux_error,
                    abs(float(row["F_left"]) - faces[0]),
                    abs(float(row["F_right"]) - faces[-1]))
                near(faces[0], 0.5, "physical left boundary flux", 3e-13)
                near(faces[-1], 0.5, "physical right boundary flux", 3e-13)
                near(row["F_left"], faces[0], "replayed left flux", 3e-13)
                near(row["F_right"], faces[-1], "replayed right flux", 3e-13)
                m_before, e_before = DX * float(np.sum(u)), energy(u)
                new = np.array([
                    u[j] - dt/DX * (faces[j+1] - faces[j])
                    for j in range(N)], dtype=np.float64)
                m_after, e_after = DX * float(np.sum(new)), energy(new)
                increment = dt * (faces[0] - faces[-1])
                budget_integral += increment
                step_defect = (m_after - m_before) - increment
                defect = (m_after - m0) - budget_integral
                max_budget_error = max(max_budget_error, abs(step_defect), abs(defect))
                near(row["mass_before"], m_before, "step mass before")
                near(row["mass_after"], m_after, "step mass after")
                near(row["flux_budget_increment"], increment, "step boundary contribution")
                near(row["flux_budget_cumulative"], budget_integral, "cumulative boundary integral")
                near(row["step_mass_defect"], step_defect, "step balance defect")
                near(row["mass_defect"], defect, "total balance defect")
                near(row["entropy_before"], e_before, "entropy before")
                near(row["entropy_after"], e_after, "entropy after")
                near(row["entropy_balance"],
                     e_after-e0+(2.0/3.0)*(t+dt), "entropy balance at step")
                near(row["min"], new.min(), "step minimum")
                near(row["max"], new.max(), "step maximum")
                near(row["total_variation"], variation(new), "step TV")
                require(np.isfinite(new).all() and new.min() >= -1.0-1e-12
                        and new.max() <= 1.0+1e-12, "range/nonfinite error")
                require(variation(new) <= previous_tv + 2e-12, "unexpected TV increase")
                previous_tv = variation(new)
                t += dt
                near(row["time_end"], t, "step end time", 3e-14)
                require(t <= target+3e-14, "checkpoint overshoot")
                u = new
                count += 1
                last_dt = dt
                dt_sequence.append(dt)
            near(t, target, "unreached checkpoint", 3e-14)
            t = target
            frame = snaps[(imethod*4+it)*N:(imethod*4+it+1)*N]
            require([int(x["i"]) for x in frame] == list(range(N)),
                    "snapshot cell IDs wrong")
            observed = np.array([float(x["U"]) for x in frame])
            exact = independent_integral_reference(target)
            official_ref = np.array([float(x["U_exact"]) for x in frame])
            err = float(np.max(abs(observed-u)))
            max_replay_state_error = max(max_replay_state_error, err)
            require(np.isfinite(observed).all() and err < 2e-12,
                    "independent scalar replay diverged")
            require(float(np.max(abs(official_ref-exact))) < 2e-12,
                    "exact reference is not integral cell average")
            for j, row in enumerate(frame):
                xl = -1.0 + j*DX
                require(row["method"] == method and float(row["time"]) == target,
                        "snapshot method or time mismatch")
                near(row["x_left"], xl, "cell left")
                near(row["x_right"], xl + DX, "cell right")
                near(row["x_center"], xl + 0.5*DX, "cell center")
            m = method_rows[it]
            require(m["experiment_id"] == "ET005-03" and
                    m["case_id"] == "rarefaction_m1_1" and
                    m["status"] == "computed_unvalidated" and
                    m["boundary"] == "riemann" and
                    m["TV_boundary"] == "riemann_fixed_exterior" and
                    m["reference_kind"] == "exact_rarefaction_cellavg",
                    "metrics scientific identity wrong")
            near(m["N"], N, "mesh N", 0)
            near(m["dx"], DX, "mesh dx", 1e-15)
            near(m["cfl_requested"], CFL, "requested CFL", 0)
            require(int(m["steps"]) == count, "checkpoint step count wrong")
            if it == 0:
                require(m["dt_last"] == "", "nonzero time step at t=0")
            else:
                near(m["dt_last"], last_dt, "last dt", 3e-14)
            mass = DX * float(np.sum(observed))
            e = energy(observed)
            l1 = DX * float(np.abs(observed - exact).sum())
            entropy_balance = e-e0+(2.0/3.0)*target
            near(m["initial_mass"], m0, "initial mass")
            near(m["mass"], mass, "measured mass")
            near(m["exact_mass"], 0.0, "analytic mass", 0)
            near(m["net_boundary_flux_integral"], budget_integral, "net boundary integral")
            near(m["mass_defect"], mass-m0-budget_integral, "mass defect")
            require(abs(mass) < 1e-11 and abs(mass-m0-budget_integral) < 1e-11,
                    "equal exterior fluxes must conserve domain mass")
            near(m["min"], observed.min(), "minimum")
            near(m["max"], observed.max(), "maximum")
            near(m["total_variation"], variation(observed), "total variation")
            near(m["L1_cellavg"], l1, "exact cell-average L1")
            near(m["energy"], e, "quadratic entropy")
            near(m["entropy_boundary_integral"], (2.0/3.0)*target,
                 "physical entropy boundary integral")
            near(m["entropy_balance"], entropy_balance, "global entropy inequality")
            if method == "Roe_no_entropy_fix":
                stationarity = float(np.max(abs(observed-init)))
                near(m["stationary_max_difference"], stationarity, "Roe stationarity")
                require(stationarity < 1e-12, "wrong: Roe stationary jump has moved")
                require(abs(l1-target) < 3e-12,
                        "static Roe exact-cell-average L1 must equal time")
                if target > 0:
                    require(entropy_balance > 0.0, "Roe nonentropic jump was hidden")
                    near(entropy_balance, (2.0/3.0)*target, "Roe entropy violation")
            else:
                require(m["stationary_max_difference"] == "",
                        "non-Roe stationarity incorrectly reported")
                require(entropy_balance <= 1e-10,
                        "Godunov violates global entropy inequality")
            if target == 0.5:
                results[method] = {
                    "L1_cellavg_T": l1, "mass_T": mass,
                    "entropy_balance_T": entropy_balance,
                    "stationary_max_difference_T": (stationarity
                          if method == "Roe_no_entropy_fix" else None)
                }
        require(count == 64, "wrong method step count")
        dt_sequences[method] = dt_sequence
    require(len(dt_sequences["Godunov"]) == len(dt_sequences["Roe_no_entropy_fix"])
            and all(abs(a-b) < 1e-15 for a,b in
                    zip(dt_sequences["Godunov"], dt_sequences["Roe_no_entropy_fix"])),
            "the two methods used different timestep sequences")
    require(results["Godunov"]["L1_cellavg_T"] <
            results["Roe_no_entropy_fix"]["L1_cellavg_T"],
            "entropy-correct Godunov did not approach the reference")
    require(abs(results["Roe_no_entropy_fix"]["L1_cellavg_T"] - 0.5) < 3e-12,
            "Roe's exact-cell-average counterexample was not preserved")
    require(max_budget_error < 1e-11 and max_flux_error < 1e-12
            and max_replay_state_error < 2e-12, "hard numerical QA failed")
    return {
        "status": "PASS", "experiment_id": "ET005-03",
        "source_git_sha": info["source_git_sha"],
        "config_sha256": info["config_sha256"],
        "design_sha256": info["design_sha256"],
        "verified_times": list(TIMES), "methods": list(METHODS),
        "steps_per_method": 64, "snapshot_rows": len(snaps),
        "metric_rows": len(metrics), "step_budget_rows": len(ledger),
        "max_mass_budget_defect": max_budget_error,
        "max_boundary_flux_replay_error": max_flux_error,
        "max_independent_FVM_state_error": max_replay_state_error,
        "theoretical_stationary_entropy_jump_violation": entropy_jump_defect,
        "T0_5": results,
        "independent_verifier": "scalar_201_face_replay_per_step_plus_piecewise_integral_reference",
        "output_sha256": info["output_sha256"]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--design", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    path = args.out / "checks.json"
    try:
        result = audit(args.out, args.config, args.design)
    except Exception as exc:
        result = {"status": "FAIL", "experiment_id": "ET005-03",
                  "failed_reason": f"{type(exc).__name__}: {exc}"}
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(result["failed_reason"], file=sys.stderr)
        return 1
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
