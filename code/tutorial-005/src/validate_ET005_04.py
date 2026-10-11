"""Independent fail-closed ET005-04 validator, no import of tested solver/refs.

Every Godunov face on every time step is independently rebuilt using a scalar
formula. Exact entropy cell means use piecewise integrals rather than the
production antiderivatives. Registered N=200 trajectories are independently
matched against the SHA-pinned, already frozen ET005-02/03 Godunov archives.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

CASE_STATES = {"shock_1_0": (1.0, 0.0),
               "rarefaction_m1_1": (-1.0, 1.0)}
CASES = ("shock_1_0", "rarefaction_m1_1")
NS = (100, 200, 400, 800)
TIMES = (0.0, 0.1, 0.25, 0.5)
CFL = 0.8
CSV_NAMES = ("grid_errors.csv", "snapshots.csv", "step_budget.csv")
GRID_COLS = (
    "experiment_id", "case_id", "method", "boundary", "N", "dx",
    "cfl_requested", "time", "steps", "dt_last", "initial_mass", "mass",
    "exact_mass", "net_boundary_flux_integral", "mass_defect",
    "minimum", "maximum", "total_variation", "TV_boundary",
    "L1_cellavg", "reference_kind", "previous_N", "p_obs_from_previous",
    "shock_speed_exact", "shock_x_exact", "shock_x_half",
    "shock_x_half_error", "status"
)
SNAP_COLS = ("case_id", "method", "N", "time", "i", "x_left",
             "x_right", "x_center", "U", "U_exact")
LEDGER_COLS = (
    "case_id", "method", "N", "step", "time_start", "time_end",
    "dt", "cfl_actual", "F_left", "F_right", "mass_before",
    "mass_after", "flux_budget_increment", "flux_budget_cumulative",
    "step_mass_defect", "mass_defect", "minimum", "maximum",
    "total_variation"
)
# The scientific identity of the source data is frozen in prior ET005-02/03.
FROZEN_ARCHIVE_SHA256 = {
    "shock_1_0": {
        "snapshots.csv": "b09f0e4184314e116acdc6b4f1606498bb53203857e01162fc4d337280182700",
        "metrics.csv": "e06f7fbea74a43440cbfc268893c6d6064844b524887287276b3566a62bf6fa8"
    },
    "rarefaction_m1_1": {
        "snapshots.csv": "4c321f877af5d523b3e0e0ffaeec5c07a260077063109e038384396b975cde2e",
        "metrics.csv": "ec1e9500cb232a3e00e6f6c7b8bf8a3dcc91957677dd34a94584783bceb81be4"
    }
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected_columns):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(expected_columns),
                f"unexpected schema in {path}")
        return list(reader)


def near(actual, expected, name, tol=2e-12):
    a = float(actual)
    e = float(expected)
    require(math.isfinite(a) and math.isfinite(e) and abs(a-e) <= tol,
            f"{name}: measured {a}, expected {e} (tol={tol})")


def godunov_scalar(a, b):
    fa, fb = a*a/2.0, b*b/2.0
    if a > b:
        return max(fa, fb)
    if a >= 0:
        return fa
    if b <= 0:
        return fb
    return 0.0


def reference_integral(case, n, t=0.5):
    """Integrate left constants / fan / right constants directly on each cell."""
    dx = 2.0/n
    output = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, b = -1.0+i*dx, -1.0+(i+1)*dx
        if case == "shock_1_0":
            shock_position = .5*t
            output[i] = max(0.0, min(b, shock_position)-a)/dx
        elif case == "rarefaction_m1_1":
            left_segment = max(0.0, min(b, -t)-a)
            right_segment = max(0.0, b-max(a, t))
            mid_left, mid_right = max(a, -t), min(b, t)
            fan = ((mid_right**2-mid_left**2)/(2.0*t)
                   if mid_right>mid_left else 0.0)
            output[i] = (-left_segment+fan+right_segment)/dx
        else:
            raise ValueError("unregistered case")
    return output


def variation(u, left, right):
    return (float(np.abs(np.diff(u)).sum())
            +abs(float(u[0]-left))+abs(float(right-u[-1])))


def xhalf_shock(u, dx):
    """Position determined only by the profile: unique decreasing .5 crossing."""
    crossings = [i for i in range(len(u)-1)
                 if float(u[i]) >= .5 > float(u[i+1])]
    require(len(crossings)==1, "shock has no unique half-value crossing")
    i=crossings[0]
    delta=float(u[i]-u[i+1])
    require(delta>0, "invalid decreasing shock")
    return -1.0+(i+.5)*dx+dx*(float(u[i])-.5)/delta


def historical_check(case, observed, measured_l1, archive_root):
    """Compare each N=200 terminal profile and L1 with prior immutable ET archives."""
    eid = "ET005-02" if case=="shock_1_0" else "ET005-03"
    directory = archive_root/eid
    for fn, digest in FROZEN_ARCHIVE_SHA256[case].items():
        require(sha(directory/fn)==digest,
                f"{eid} previously frozen archive identity changed: {fn}")
    history = read_untyped(directory/"snapshots.csv")
    historical = [r for r in history
                  if float(r["time"])==.5 and
                  (eid=="ET005-02" or r["method"]=="Godunov")]
    require(len(historical)==200 and
            [int(r["i"]) for r in historical]==list(range(200)),
            "missing registered archived N200 Godunov terminal profile")
    archived_u = np.array([float(r["U"]) for r in historical],dtype=np.float64)
    max_state = float(np.max(np.abs(observed-archived_u)))
    require(max_state < 3e-13,
            f"{eid} independent frozen N200 state mismatch {max_state}")
    original_metrics = read_untyped(directory/"metrics.csv")
    matched = [r for r in original_metrics
               if float(r["time"])==.5 and
               (eid=="ET005-02" or r["method"]=="Godunov")]
    require(len(matched)==1, "ambiguous archived final Godunov metric")
    archived_l1 = float(matched[0]["L1_cellavg"])
    diff=abs(measured_l1-archived_l1)
    require(diff < 3e-13, f"{eid} N200 L1 mismatch {diff}")
    return {"archive": eid,
            "max_abs_terminal_U_difference": max_state,
            "abs_terminal_L1_difference": diff,
            "old_L1_cellavg": archived_l1}


def read_untyped(path):
    with Path(path).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def audit(output, config_path, design_path, archive_root):
    out=Path(output)
    config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    design=json.loads(Path(design_path).read_text(encoding="utf-8"))
    require(config == {
        "contract_version":"v0.2-frozen-2026-10-10",
        "experiment_id":"ET005-04","method":"Godunov",
        "boundary":"riemann","domain":[-1,1],
        "cases":list(CASES),"N":list(NS),
        "requested_cfl":.8,"a_bound":1,
        "times":list(TIMES),
        "reference_kind":"exact_entropy_solution_cellavg",
        "output_snapshot_time":.5
    },"ET005-04 registered matrix or output schedule changed")
    require(design["contract_version"]==config["contract_version"]
            and design["ET005-04_N"]==list(NS)
            and design["riemann_cases"]["domain"]==[-1,1]
            and design["riemann_cases"]["base_N"]==200
            and design["riemann_cases"]["final_time"]==.5
            and design["riemann_cases"]["shock_1_0"]==[1.,0.]
            and design["riemann_cases"]["rarefaction_m1_1"]==[-1.,1.]
            and design["base_requested_cfl"]==.8
            and design["wave_speed_bound"]==1.0,
            "shared v0.2 design mutated")
    info=json.loads((out/"run.json").read_text(encoding="utf-8"))
    require(info["experiment_id"]=="ET005-04" and
            info["status"]=="computed_unvalidated" and
            isinstance(info["source_git_sha"],str) and bool(info["source_git_sha"]),
            "invalid raw run status/provenance")
    require(info["config_sha256"]==sha(config_path)
            and info["design_sha256"]==sha(design_path),
            "input SHA256 differs from registered inputs")
    require(info["archive_cases"]==list(CASES)
            and info["archive_N"]==list(NS)
            and info["checkpoints"]==list(TIMES)
            and info["terminal_time"]==.5,
            "invalid run settings or archive grid")
    require(set(info["output_sha256"])==set(CSV_NAMES)
            and set(info["output_counts"])==set(CSV_NAMES),
            "missing output identity")
    for fn in CSV_NAMES:
        require(sha(out/fn)==info["output_sha256"][fn],
                "changed generated CSV SHA256: "+fn)
    rows=read(out/"grid_errors.csv",GRID_COLS)
    snaps=read(out/"snapshots.csv",SNAP_COLS)
    ledger=read(out/"step_budget.csv",LEDGER_COLS)
    require(len(rows)==8 and len(snaps)==3000,
            "missing 2x4 grid or final-time snapshots")
    require(info["output_counts"]=={"grid_errors.csv":8,
            "snapshots.csv":3000,"step_budget.csv":len(ledger)},
            "run metadata cardinality mismatch")
    require([(x["case_id"],int(x["N"])) for x in rows] ==
            [(case,n) for case in CASES for n in NS],
            "incomplete, duplicated or reordered 2x4 registered grid")
    require({(x["case_id"],int(x["N"])) for x in ledger} ==
            {(case,n) for case in CASES for n in NS},
            "missing or unregistered ledger group")
    all_errors={}
    crosschecks={}
    summary=[]
    snap_idx=0
    max_state_error=0.0
    max_mass_error=0.0
    max_boundary_error=0.0
    total_steps=0
    for case in CASES:
        ul,ur=CASE_STATES[case]
        prev_n=None
        prev_l1=None
        for n in NS:
            dx=2.0/n
            # Reconstruct the *stored floating-point* integral means
            # independently from the same registered geometric faces.
            # numpy.linspace faces have roundoff-level cell-width variation;
            # forcing mathematical +/-1 here would introduce spurious TV
            # discrepancies, especially after 250 steps on N=800.
            faces0=np.linspace(-1.0,1.0,n+1,dtype=np.float64)
            left_lengths=np.array([
                max(0.0,min(dx,min(float(faces0[i+1]),0.0)-float(faces0[i])))
                for i in range(n)],dtype=np.float64)
            u=np.array([
                (ul*float(ll)+ur*(dx-float(ll)))/dx
                for ll in left_lengths],dtype=np.float64)
            ideal=np.array([ul]*(n//2)+[ur]*(n//2),dtype=np.float64)
            require(float(np.max(np.abs(u-ideal)))<5e-12,
                    "real initial cell integrals are not equal to registered step")
            m0=dx*float(np.sum(u))
            t=0.0
            count=0
            last_dt=None
            cumulative=0.0
            previous_tv=variation(u,ul,ur)
            group=[x for x in ledger if x["case_id"]==case and int(x["N"])==n]
            for target in TIMES:
                while target-t > 2e-14:
                    require(count < len(group),"incomplete step ledger")
                    row=group[count]
                    require(row["case_id"]==case and row["method"]=="Godunov"
                            and int(row["N"])==n and int(row["step"])==count+1,
                            "ledger index or method mutated")
                    near(row["time_start"],t,"checkpoint scheduler time start",3e-14)
                    dt=min(.8*dx,target-t)
                    require(dt>0.0 and dt/dx <= 1.0,
                            "nonpositive or CFL-violating time step")
                    near(row["dt"],dt,"frozen dt history",3e-14)
                    near(row["cfl_actual"],dt/dx,"actual CFL",3e-13)
                    faces=[godunov_scalar(ul,float(u[0]))]
                    for j in range(1,n):
                        faces.append(godunov_scalar(float(u[j-1]),float(u[j])))
                    faces.append(godunov_scalar(float(u[-1]),ur))
                    expected_fl=.5
                    expected_fr=0.0 if case=="shock_1_0" else .5
                    near(faces[0],expected_fl,"left exterior flux",3e-13)
                    near(faces[-1],expected_fr,"right exterior flux",3e-13)
                    max_boundary_error=max(max_boundary_error,
                        abs(float(row["F_left"])-faces[0]),
                        abs(float(row["F_right"])-faces[-1]))
                    near(row["F_left"],faces[0],"registered left boundary face")
                    near(row["F_right"],faces[-1],"registered right boundary face")
                    mbefore=dx*float(np.sum(u))
                    new=np.array([u[j]-(dt/dx)*(faces[j+1]-faces[j])
                                  for j in range(n)],dtype=np.float64)
                    mafter=dx*float(np.sum(new))
                    inc=dt*(faces[0]-faces[-1])
                    cumulative+=inc
                    step_defect=(mafter-mbefore)-inc
                    defect=(mafter-m0)-cumulative
                    max_mass_error=max(max_mass_error,abs(step_defect),abs(defect))
                    near(row["mass_before"],mbefore,"mass before")
                    near(row["mass_after"],mafter,"mass after")
                    near(row["flux_budget_increment"],inc,"step boundary increment")
                    near(row["flux_budget_cumulative"],cumulative,
                         "cumulative flux budget")
                    near(row["step_mass_defect"],step_defect,"step mass defect")
                    near(row["mass_defect"],defect,"cumulative mass defect")
                    near(row["minimum"],new.min(),"step min")
                    near(row["maximum"],new.max(),"step max")
                    tv=variation(new,ul,ur)
                    near(row["total_variation"],tv,"step TV")
                    require(tv <= previous_tv+2e-12,"TV monotonicity violated")
                    require(np.isfinite(new).all()
                            and new.min()>=min(ul,ur)-1e-12
                            and new.max()<=max(ul,ur)+1e-12,
                            "domain bounds/nonfinite violation")
                    previous_tv=tv
                    t+=dt
                    near(row["time_end"],t,"time end",3e-14)
                    require(t<=target+3e-14,"overshot checkpoint")
                    u=new
                    count+=1
                    last_dt=dt
                near(t,target,"did not reach registered sample",3e-14)
                t=target
            require(len(group)==count,"extra step ledger rows")
            total_steps+=count
            frame=snaps[snap_idx:snap_idx+n]
            snap_idx+=n
            require(len(frame)==n and
                    [int(x["i"]) for x in frame]==list(range(n)),
                    "missing final snapshot cell ID")
            observed=np.array([float(x["U"]) for x in frame],dtype=np.float64)
            independent_exact=reference_integral(case,n)
            recorded_exact=np.array([float(x["U_exact"]) for x in frame])
            max_state_error=max(max_state_error,
                                float(np.max(np.abs(observed-u))))
            require(np.max(np.abs(observed-u))<3e-12,
                    "scalar every-face replay differs from production")
            require(np.max(np.abs(recorded_exact-independent_exact))<3e-12,
                    "reference differs from independent piecewise integrals")
            for i,x in enumerate(frame):
                xl=-1.0+i*dx
                require(x["case_id"]==case and x["method"]=="Godunov"
                        and int(x["N"])==n and float(x["time"])==.5,
                        "snapshot identity not frozen")
                near(x["x_left"],xl,"left face",3e-14)
                near(x["x_right"],xl+dx,"right face",3e-14)
                near(x["x_center"],xl+dx/2,"cell center",3e-14)
            item=next(x for x in rows if x["case_id"]==case and int(x["N"])==n)
            require(item["experiment_id"]=="ET005-04"
                    and item["method"]=="Godunov"
                    and item["boundary"]=="riemann"
                    and item["TV_boundary"]=="riemann_fixed_exterior"
                    and item["status"]=="computed_unvalidated"
                    and item["reference_kind"]=="exact_entropy_solution_cellavg",
                    "result metadata changed")
            near(item["dx"],dx,"dx",2e-15)
            near(item["cfl_requested"],CFL,"CFL",0)
            near(item["time"],.5,"final time",0)
            require(int(item["steps"])==count,"step count differs from ledger")
            near(item["dt_last"],last_dt,"final dt",3e-14)
            mass=dx*float(np.sum(observed))
            mass_exact=1.25 if case=="shock_1_0" else 0.0
            require(abs(mass-mass_exact)<1e-11,
                    "analytic nonperiodic mass budget failed")
            near(item["initial_mass"],m0,"initial exact mass")
            near(item["mass"],mass,"terminal mass")
            near(item["exact_mass"],mass_exact,"analytic final mass",0)
            near(item["net_boundary_flux_integral"],cumulative,"net mass inflow")
            near(item["mass_defect"],mass-m0-cumulative,"terminal mass defect")
            l1=dx*float(np.abs(observed-independent_exact).sum())
            near(item["L1_cellavg"],l1,"exact-cell-mean L1",3e-12)
            near(item["minimum"],observed.min(),"min")
            near(item["maximum"],observed.max(),"max")
            near(item["total_variation"],variation(observed,ul,ur),"terminal TV")
            if case=="shock_1_0":
                pos=xhalf_shock(observed,dx)
                near(item["shock_speed_exact"],.5,"exact shock speed",0)
                near(item["shock_x_exact"],.25,"exact shock position",0)
                near(item["shock_x_half"],pos,"independent threshold",3e-12)
                near(item["shock_x_half_error"],pos-.25,
                     "threshold error",3e-12)
            else:
                require(all(item[k]=="" for k in ("shock_speed_exact",
                        "shock_x_exact","shock_x_half","shock_x_half_error")),
                        "rarefaction cannot report shock position")
            if prev_n is None:
                require(item["previous_N"]=="" and
                        item["p_obs_from_previous"]=="",
                        "first grid cannot have observed order")
                p=None
            else:
                require(int(item["previous_N"])==prev_n,
                        "observed order from wrong coarse grid")
                prior_item=next(x for x in rows
                                if x["case_id"]==case and int(x["N"])==prev_n)
                # First independently validate both L1 cell integrals (above);
                # then recompute the published order from the *published* CSV
                # values. An O(1e-14) difference between independently
                # accumulated L1 values can amplify under logarithms.
                prior_csv=float(prior_item["L1_cellavg"])
                current_csv=float(item["L1_cellavg"])
                p=(math.log(prior_csv/current_csv,2.0) if
                   prior_csv>0 and current_csv>0 and
                   math.isfinite(prior_csv) and math.isfinite(current_csv) else None)
                p_independent=(math.log(prev_l1/l1,2.0) if
                    prev_l1>0 and l1>0 and
                    math.isfinite(prev_l1) and math.isfinite(l1) else None)
                if p is None:
                    require(item["p_obs_from_previous"]=="",
                            "invalid finite observed convergence exponent")
                else:
                    near(item["p_obs_from_previous"],p,"published L1 observed order",3e-13)
                    require(p_independent is not None
                            and abs(p-p_independent)<1e-9,
                            "independent integral order mismatch")
            if n==200:
                crosschecks[case]=historical_check(
                    case,observed,l1,Path(archive_root))
            summary.append({"case_id":case,"N":n,"dx":dx,"L1_cellavg":l1,
                            "p_obs":p,"steps":count,"mass":mass,
                            "mass_defect":mass-m0-cumulative,
                            "shock_x_half":(xhalf_shock(observed,dx)
                                  if case=="shock_1_0" else None)})
            all_errors[(case,n)]=l1
            prev_n,prev_l1=n,l1
    require(snap_idx==len(snaps),"extra snapshot rows")
    require(len(ledger)==total_steps,"extra ledger group not replayed")
    require(max_mass_error<1e-11 and max_boundary_error<3e-12
            and max_state_error<3e-12,
            "global independent numerical verification failed")
    require(set(crosschecks)==set(CASES),"missing N200 official archive matching")
    return {
        "status":"PASS","experiment_id":"ET005-04",
        "source_git_sha":info["source_git_sha"],
        "config_sha256":info["config_sha256"],
        "design_sha256":info["design_sha256"],
        "grid_rows":len(rows),
        "snapshot_rows":len(snaps),
        "ledger_rows":len(ledger),
        "verified_pairs":summary,
        "N200_frozen_crosschecks":crosschecks,
        "max_independent_FVM_state_error":max_state_error,
        "max_step_and_cumulative_mass_defect":max_mass_error,
        "max_boundary_flux_replay_error":max_boundary_error,
        "independent_reference":"piecewise_geometric_integral_for_shock_and_fan",
        "independent_fvm":"scalar_face_by_face_all_steps",
        "output_sha256":info["output_sha256"]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--design",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--archive-root",type=Path,required=True)
    args=p.parse_args()
    try:
        report=audit(args.out,args.config,args.design,args.archive_root)
    except Exception as exc:
        report={"status":"FAIL","experiment_id":"ET005-04",
                "failed_reason":f"{type(exc).__name__}: {exc}"}
        (args.out/"checks.json").write_text(
            json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print(report["failed_reason"],file=sys.stderr)
        return 1
    (args.out/"checks.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=="__main__":
    sys.exit(main())
