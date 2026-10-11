"""Independent fail-closed ET005-02 verifier: no import of the tested solver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

TIMES = (0.0, 0.1, 0.25, 0.5)
N = 200
DX = 2.0 / N
REQUESTED_CFL = 0.8
U_LEFT, U_RIGHT = 1.0, 0.0
SPEED = 0.5


def require(test, reason):
    if not test:
        raise ValueError(reason)


def read_csv(path, expected_columns):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(expected_columns), "incorrect CSV schema: " + str(path))
        return list(reader)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_godunov(a, b):
    """Scalar convex Burgers Riemann flux, independent of burgers_flux.py."""
    left, right = a*a/2.0, b*b/2.0
    if a > b:
        return left if left >= right else right
    if a >= 0.0:
        return left
    if b <= 0.0:
        return right
    return 0.0


def independent_exact_means(t):
    """Integrate the indicator x<.5t through a clipped antiderivative."""
    s = 0.5*t
    grid = [-1.0 + DX*j for j in range(N+1)]
    return np.array([(min(grid[i+1],s) - min(grid[i],s))/DX
                     for i in range(N)], dtype=np.float64)


def independent_x_half(u):
    # Do not import tested position diagnostic or choose the candidate nearest s(t).
    indices = [i for i in range(N-1) if u[i] >= .5 > u[i+1]]
    require(len(indices) == 1, "not a unique descending threshold crossing")
    i = indices[0]
    denom = float(u[i]-u[i+1])
    require(denom > 0.0, "crossing has zero/negative slope")
    return -1.0 + (i+.5)*DX + DX * (float(u[i])-.5)/denom


def audit(output, config, design):
    out = Path(output)
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    common = json.loads(Path(design).read_text(encoding="utf-8"))
    require(cfg["contract_version"] == "v0.2-frozen-2026-10-10", "contract changed")
    require(cfg["experiment_id"] == "ET005-02" and cfg["case_id"] == "shock_1_0", "wrong experiment")
    require(cfg["method"] == "Godunov" and cfg["boundary"] == "riemann", "wrong flux/boundary")
    require(cfg["domain"] == [-1.0,1.0] and cfg["N"] == N, "wrong grid/domain")
    require(cfg["u_L"] == U_LEFT and cfg["u_R"] == U_RIGHT, "wrong Riemann states")
    require(cfg["requested_cfl"] == REQUESTED_CFL and cfg["a_bound"] == 1.0, "wrong CFL/bound")
    require(cfg["times"] == list(TIMES) and cfg["reference_kind"] == "exact_shock_cellavg", "wrong samples")
    require(common["base_requested_cfl"] == REQUESTED_CFL
            and common["riemann_cases"]["shock_1_0"] == [1.0,0.0]
            and common["riemann_cases"]["base_N"] == N
            and common["riemann_cases"]["final_time"] == TIMES[-1], "common design mismatch")
    info = json.loads((out/"run.json").read_text(encoding="utf-8"))
    require(info["status"] == "computed_unvalidated" and info["experiment_id"] == "ET005-02", "false driver status")
    require(info["config_sha256"] == hash_file(config), "config SHA256 mismatch")
    require(info["design_sha256"] == hash_file(design), "common design SHA256 mismatch")
    require(isinstance(info["source_git_sha"], str) and bool(info["source_git_sha"]), "no source ID")
    require(info["requested_times"] == list(TIMES), "wrong run metadata times")
    expected_outputs = ("snapshots.csv", "metrics.csv", "step_budget.csv")
    require(set(info["output_sha256"]) == set(expected_outputs), "missing/extra data files")
    for name in expected_outputs:
        require(hash_file(out/name) == info["output_sha256"][name], "bad recorded SHA: " + name)
    snapshot_cols = ("time", "i", "x_left", "x_right", "x_center", "U", "U_exact")
    metric_cols = (
        "experiment_id", "case_id", "method", "boundary", "N", "dx", "cfl_requested",
        "time", "dt_last", "steps", "initial_mass", "mass", "exact_mass",
        "net_boundary_flux_integral", "mass_defect", "min", "max", "total_variation",
        "TV_boundary", "L1_cellavg", "reference_kind", "shock_speed_exact",
        "shock_x_exact", "shock_x_half", "shock_x_half_error", "shock_x_mass", "status")
    ledger_cols = (
        "step", "time_start", "time_end", "dt", "cfl_actual", "F_left", "F_right",
        "mass_before", "mass_after", "flux_budget_increment", "flux_budget_cumulative",
        "step_mass_defect", "mass_defect", "min", "max", "total_variation")
    snaps = read_csv(out/"snapshots.csv", snapshot_cols)
    metrics = read_csv(out/"metrics.csv", metric_cols)
    ledger = read_csv(out/"step_budget.csv", ledger_cols)
    require(len(snaps) == N * len(TIMES) and len(metrics) == len(TIMES), "wrong row counts")
    require(info["output_counts"] == {"snapshots.csv":len(snaps),"metrics.csv":len(metrics),
                                      "step_budget.csv":len(ledger)}, "wrong row identity")
    require([float(row["time"]) for row in metrics] == list(TIMES), "missing or duplicated checkpoint")
    # Exact integral of initial Heaviside profile; no use of core initial_means.
    u = np.array([1.0 if i < N//2 else 0.0 for i in range(N)], dtype=np.float64)
    m0 = DX * float(np.sum(u))
    require(m0 == 1.0, "initial integral wrong")
    count, current_time, net_integral = 0, 0.0, 0.0
    max_step_defect = 0.0
    max_boundary_flux_error = 0.0
    L1_at_end = None
    max_state_discrepancy = 0.0
    for k, target in enumerate(TIMES):
        if k > 0:
            while target - current_time > 2e-14:
                require(count < len(ledger), "missing step budget")
                row = ledger[count]
                dt = min(DX*REQUESTED_CFL, target-current_time)
                recorded_dt = float(row["dt"])
                require(int(row["step"]) == count+1, "nonconsecutive step index")
                require(abs(float(row["time_start"]) - current_time) < 3e-14, "disconnected step start")
                require(abs(dt-recorded_dt) < 3e-14 and recorded_dt > 0, "wrong time scheduler")
                require(abs(float(row["cfl_actual"])-recorded_dt/DX) < 3e-13, "wrong CFL")
                require(0 < recorded_dt/DX <= 1.0 and recorded_dt <= DX*REQUESTED_CFL+3e-14,
                        "CFL/out-of-domain step")
                left_faces = np.empty(N+1, dtype=np.float64)
                left_faces[0] = scalar_godunov(U_LEFT, float(u[0]))
                for j in range(1,N):
                    left_faces[j] = scalar_godunov(float(u[j-1]),float(u[j]))
                left_faces[N] = scalar_godunov(float(u[-1]),U_RIGHT)
                max_boundary_flux_error = max(max_boundary_flux_error,
                    abs(float(row["F_left"])-left_faces[0]),
                    abs(float(row["F_right"])-left_faces[-1]))
                require(abs(float(row["F_left"])-left_faces[0]) < 3e-14, "left exterior flux mismatch")
                require(abs(float(row["F_right"])-left_faces[-1]) < 3e-14, "right exterior flux mismatch")
                require(abs(left_faces[0]-.5)<3e-14 and abs(left_faces[-1])<3e-14,
                        "outer entropy state trace invalid")
                m_before = DX * float(np.sum(u))
                updated = np.array([u[j]-recorded_dt/DX*(left_faces[j+1]-left_faces[j])
                                    for j in range(N)], dtype=np.float64)
                m_after = DX * float(np.sum(updated))
                delta = recorded_dt*(left_faces[0]-left_faces[-1])
                net_integral += delta
                defect_step = (m_after-m_before)-delta
                defect_total = (m_after-m0)-net_integral
                max_step_defect=max(max_step_defect,abs(defect_step),abs(defect_total))
                require(abs(m_before-float(row["mass_before"])) < 2e-12,"wrong mass before")
                require(abs(m_after-float(row["mass_after"])) < 2e-12,"wrong mass after")
                require(abs(delta-float(row["flux_budget_increment"])) < 3e-14,"wrong flux increment")
                require(abs(net_integral-float(row["flux_budget_cumulative"])) < 2e-12,"wrong flux cumulative")
                require(abs(defect_step-float(row["step_mass_defect"])) < 2e-12,"wrong step defect")
                require(abs(defect_total-float(row["mass_defect"])) < 2e-12,"wrong total defect")
                require(np.isfinite(updated).all() and updated.min()>=-1e-12 and updated.max()<=1.+1e-12,
                        "finite-value/maximum principle violated")
                tv = float(np.abs(np.diff(updated)).sum()+abs(updated[0]-U_LEFT)+abs(U_RIGHT-updated[-1]))
                require(abs(tv-float(row["total_variation"])) < 2e-12,"wrong TV")
                require(abs(float(row["min"])-updated.min())<2e-12 and abs(float(row["max"])-updated.max())<2e-12,
                        "wrong reported range")
                current_time += recorded_dt
                require(abs(current_time-float(row["time_end"])) < 3e-14,"disconnected step end")
                require(current_time <= target + 3e-14, "time overshot checkpoint")
                u=updated
                count+=1
            require(abs(current_time-target)<3e-14, "unreached checkpoint")
            current_time=target
        frame = snaps[k*N:(k+1)*N]
        require(len(frame)==N and [int(row["i"]) for row in frame]==list(range(N)),
                "snapshot row order or IDs invalid")
        raw=np.array([float(row["U"]) for row in frame],dtype=np.float64)
        exact=independent_exact_means(target)
        require(np.isfinite(raw).all() and np.max(abs(raw-u)) < 2e-13,"snapshot differs from independent replay")
        max_state_discrepancy=max(max_state_discrepancy,float(np.max(abs(raw-u))))
        require(np.max(abs(np.array([float(row["U_exact"]) for row in frame])-exact))<3e-14,
                "wrong exact per-cell integral")
        for j,row in enumerate(frame):
            xl=-1.0+j*DX
            require(float(row["time"])==target and abs(float(row["x_left"])-xl)<3e-14,
                    "wrong cell location/time")
            require(abs(float(row["x_right"])-(xl+DX))<3e-14 and
                    abs(float(row["x_center"])-(xl+.5*DX))<3e-14,
                    "wrong Riemann cell faces/centers")
        m=metrics[k]
        require(m["status"]=="computed_unvalidated" and m["method"]=="Godunov" and
                m["case_id"]=="shock_1_0" and m["boundary"]=="riemann", "wrong metric identity")
        require(m["TV_boundary"]=="riemann_fixed_exterior" and
                m["reference_kind"]=="exact_shock_cellavg", "wrong metric reference/TV")
        require(int(m["steps"])==count and int(m["N"])==N and float(m["dx"])==DX,
                "wrong mesh or step count")
        require(float(m["cfl_requested"])==REQUESTED_CFL,"wrong requested CFL")
        require((m["dt_last"]=="") if k==0 else
                abs(float(m["dt_last"])-float(ledger[count-1]["dt"])) < 3e-14,
                "wrong final step record")
        mass=DX*float(np.sum(raw))
        expected_mass=1.0+.5*target
        mass_defect=(mass-m0)-net_integral
        L1=DX*float(np.abs(raw-exact).sum())
        x_half=independent_x_half(raw)
        shock_exact=.5*target
        require(abs(float(m["initial_mass"])-m0)<3e-14,"wrong initial mass")
        require(abs(float(m["mass"])-mass)<2e-12,"wrong reported mass")
        require(abs(float(m["exact_mass"])-expected_mass)<2e-14,"wrong analytical mass")
        require(abs(mass-expected_mass) < 1e-11,"physical nonperiodic mass failed")
        require(abs(float(m["net_boundary_flux_integral"])-net_integral)<2e-12,"wrong boundary integral")
        require(abs(float(m["mass_defect"])-mass_defect)<2e-12 and abs(mass_defect)<1e-11,
                "mass defect beyond tolerance")
        require(abs(float(m["L1_cellavg"])-L1)<2e-13,"wrong exact cell-average L1")
        require(abs(float(m["shock_speed_exact"])-SPEED)<2e-14,"wrong RH shock speed")
        require(abs(float(m["shock_x_exact"])-shock_exact)<2e-14,"wrong theoretical shock location")
        require(abs(float(m["shock_x_half"])-x_half)<3e-13,"wrong independent crossing")
        require(abs(float(m["shock_x_half_error"])-(x_half-shock_exact))<3e-13,
                "wrong threshold crossing error")
        require(abs(float(m["shock_x_mass"])-(-1.0+mass))<3e-13,
                "wrong secondary mass-inferred position")
        require(abs(float(m["total_variation"])-
                    (float(np.abs(np.diff(raw)).sum())+abs(float(raw[0]-1.0))+abs(float(raw[-1]))))<2e-12,
                "wrong nonperiodic TV")
        require(abs(float(m["min"])-raw.min())<2e-12 and abs(float(m["max"])-raw.max())<2e-12,
                "wrong range")
        if k==len(TIMES)-1:
            L1_at_end=L1
    require(count==len(ledger),"extra/missing ledger rows")
    require(abs(current_time-TIMES[-1])<3e-14, "did not reach terminal time")
    require(max_step_defect<1e-11 and max_boundary_flux_error<1e-13,
            "independent per-step budget failed")
    return {
        "status":"PASS", "experiment_id":"ET005-02",
        "source_git_sha":info["source_git_sha"],
        "config_sha256": info["config_sha256"],
        "design_sha256":info["design_sha256"],
        "verified_times":list(TIMES), "actual_steps":count,
        "snapshots":len(snaps), "step_budget_rows":len(ledger),
        "analytic_shock_speed":SPEED,
        "analytic_shock_x_T":SPEED*TIMES[-1],
        "analytic_mass_T":1.0+SPEED*TIMES[-1],
        "L1_cellavg_T":L1_at_end,
        "max_step_and_cumulative_mass_defect":max_step_defect,
        "max_boundary_flux_replay_discrepancy":max_boundary_flux_error,
        "max_independent_FVM_discrepancy":max_state_discrepancy,
        "validator_independent":"scalar_godunov_each_of_N_plus_one_faces_per_step_and_exact_step_integral",
        "output_sha256":info["output_sha256"],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--design",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    target=args.out/"checks.json"
    try:
        result=audit(args.out,args.config,args.design)
    except Exception as exc:
        fail={"status":"FAIL","experiment_id":"ET005-02",
              "failed_reason":f"{type(exc).__name__}: {exc}"}
        target.write_text(json.dumps(fail,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(fail["failed_reason"],file=sys.stderr)
        return 1
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    sys.exit(main())
