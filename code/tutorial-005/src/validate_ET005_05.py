"""ET005-05 independent fail-closed validator.

Does not import the tested Burgers FV solver, vector numerical_flux, any ET05
diagnostics, or the tested exact-reference implementation. Every accepted
step is reconstructed via scalar Riemann fluxes, full Riemann boundary ledger,
geometric cell-average reference, shape and transition-width diagnostics.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

METHODS=("Godunov","LF_alpha1")
CFLS=(0.4,0.7,0.9)
TIMES=(0.0,0.1,0.25,0.5)
N=200
DX=2.0/N
EXPECTED_PAIRS=[(m,c) for m in METHODS for c in CFLS]
SNAP_COLS=("method","cfl_requested","N","time","i","x_left","x_right","x_center","U","U_exact")
LEDGER_COLS=(
    "method","cfl_requested","step","time_start","time_end","dt",
    "cfl_actual","F_left","F_right","mass_before","mass_after",
    "flux_budget_increment","flux_budget_cumulative","step_mass_defect",
    "mass_defect","min","max","total_variation"
)
METRIC_COLS=(
    "experiment_id","case_id","method","boundary","N","dx",
    "LF_alpha","cfl_requested","time","steps","dt_last",
    "initial_mass","mass","exact_mass","net_boundary_flux_integral",
    "mass_defect","min","max","total_variation","TV_boundary",
    "L1_cellavg","reference_kind","shock_speed_exact","shock_x_exact",
    "shock_x_half","shock_x_half_error","shock_x_mass",
    "transition_low","transition_high","transition_cells",
    "transition_segments","transition_width","quadratic_content","status"
)
FROZEN_ET00502_HASHES={
    "metrics.csv":"e06f7fbea74a43440cbfc268893c6d6064844b524887287276b3566a62bf6fa8",
    "snapshots.csv":"b09f0e4184314e116acdc6b4f1606498bb53203857e01162fc4d337280182700"
}


def require(condition,reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(actual,expected,label,tol=2e-12):
    a,e=float(actual),float(expected)
    require(math.isfinite(a) and math.isfinite(e) and abs(a-e)<=tol,
            f"{label}: {a} versus {e}; tolerance={tol}")


def read(path,expected=None):
    with Path(path).open(newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        if expected is not None:
            require(rd.fieldnames==list(expected),"incorrect columns: "+str(path))
        return list(rd)


def scalar_flux(a,b,method):
    """Independent scalar, not the production vectorized flux."""
    fa,fb=.5*a*a,.5*b*b
    if method=="Godunov":
        if a>b:
            return max(fa,fb)
        if a>=0:
            return fa
        if b<=0:
            return fb
        return 0.0
    if method=="LF_alpha1":
        # Fixed global alpha = 1; never max(|a|,|b|) or dynamically recomputed.
        return .5*(fa+fb)-.5*(b-a)
    raise ValueError("unregistered scalar flux method")


def integral_exact(n=N,t=.5):
    """Reference: exact length of overlap of each cell with x < s(t)."""
    speed=.5
    shock=speed*t
    dx=2.0/n
    out=np.empty(n,dtype=np.float64)
    for i in range(n):
        a=-1.+i*dx
        b=-1.+(i+1)*dx
        left_length=max(0.,min(b,shock)-a)
        out[i]=left_length/dx
    return out


def original_means(n=N):
    """Independent geometric cell integrals with the source's grid endpoints.

    Independent of production riemann_initial_means. The tiny floating-point
    width discrepancies from np.linspace are propagated, not silently clipped
    or replaced by idealized constant values before numerical replay.
    """
    grid=np.linspace(-1.0,1.0,n+1,dtype=np.float64)
    dx=2.0/n
    values=[]
    for i in range(n):
        a,b=float(grid[i]),float(grid[i+1])
        left_length=max(0.0,min(dx,min(b,0.0)-a))
        values.append((left_length+0.0*(dx-left_length))/dx)
    ans=np.array(values,dtype=np.float64)
    truth=np.array([1.0]*(n//2)+[0.0]*(n//2),dtype=np.float64)
    require(np.max(abs(ans-truth))<5e-12,"initial integral differs from exact Riemann geometry")
    return ans


def shape_diagnostics(state,dx=DX):
    u=np.asarray(state,dtype=np.float64)
    require(u.size==N and np.isfinite(u).all(),"invalid terminal vector")
    centers=-1.0+(np.arange(N,dtype=np.float64)+.5)*dx
    crossing=np.flatnonzero((u[:-1]>=.5)&(u[1:]<.5))
    require(len(crossing)==1,
            "half-height location is not a unique descending crossing")
    j=int(crossing[0])
    slope=float(u[j]-u[j+1])
    require(slope>0.0,"invalid shock slope")
    xhalf=float(centers[j]+dx*((float(u[j])-.5)/slope))
    mask=(u>.1)&(u<.9)
    n_cells=int(np.count_nonzero(mask))
    n_segments=int(bool(mask[0]))+int(np.count_nonzero(mask[1:]&(~mask[:-1])))
    tv=float(np.abs(np.diff(u)).sum()+abs(u[0]-1.)+abs(u[-1]))
    quadratic=float(.5*dx*np.dot(u,u))
    return {
        "x_half":xhalf,"transition_cells":n_cells,
        "transition_segments":n_segments,"transition_width":n_cells*dx,
        "TV":tv,"quadratic_content":quadratic
    }


def audit(out,config_path,design_path,archive_root):
    path=Path(out)
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    base=json.loads(Path(design_path).read_text(encoding="utf-8"))
    expected={
        "contract_version":"v0.2-frozen-2026-10-10",
        "experiment_id":"ET005-05","case_id":"shock_1_0",
        "methods":list(METHODS),"boundary":"riemann",
        "domain":[-1.0,1.0],"N":N,"u_L":1.0,"u_R":0.0,
        "LF_alpha":1.0,"requested_cfl":list(CFLS),"a_bound":1.0,
        "times":list(TIMES),"snapshot_time":.5,
        "reference_kind":"exact_shock_cellavg",
        "transition_level_low":.1,"transition_level_high":.9
    }
    require(cfg==expected,"frozen six-configuration ET005-05 registration changed")
    require(base["contract_version"]==cfg["contract_version"]
            and base["equation"]=="u_t+(u^2/2)_x=0"
            and base["riemann_cases"]["domain"]==[-1.,1.]
            and base["riemann_cases"]["base_N"]==200
            and base["riemann_cases"]["shock_1_0"]==[1.,0.]
            and base["riemann_cases"]["final_time"]==.5
            and base["ET005-05_methods"]==list(METHODS)
            and base["ET005-05_cfl"]==list(CFLS)
            and base["wave_speed_bound"]==1.,
            "previously frozen common design mutated")
    near(scalar_flux(1.,0.,"Godunov"),.5,"Godunov shock face",0)
    near(scalar_flux(1.,0.,"LF_alpha1"),.75,"global LF shock face",0)
    near(scalar_flux(1.,1.,"LF_alpha1"),.5,"LF constant",0)
    near(scalar_flux(0.,0.,"LF_alpha1"),0.,"LF constant zero",0)

    info=json.loads((path/"run.json").read_text(encoding="utf-8"))
    require(info["experiment_id"]=="ET005-05" and
            info["status"]=="computed_unvalidated" and
            isinstance(info["source_git_sha"],str) and info["source_git_sha"],
            "raw scientific result status/provenance invalid")
    require(info["config_sha256"]==sha(config_path) and
            info["design_sha256"]==sha(design_path),
            "config/design content hash differs")
    require(info["methods"]==list(METHODS) and
            info["cfl_requested"]==list(CFLS) and
            info["requested_times"]==list(TIMES) and
            info["alpha_LF"]==1.0,
            "registered method/CFL/timing identity mismatch")
    output_files=("cfl_flux_metrics.csv","snapshots.csv","step_budget.csv")
    require(set(info["output_sha256"])==set(output_files) and
            set(info["output_counts"])==set(output_files),
            "missing/extra raw CSV identity")
    for name in output_files:
        require(sha(path/name)==info["output_sha256"][name],
                "raw output SHA256 mismatch: "+name)
    metrics=read(path/"cfl_flux_metrics.csv",METRIC_COLS)
    snapshots=read(path/"snapshots.csv",SNAP_COLS)
    ledger=read(path/"step_budget.csv",LEDGER_COLS)
    require(len(metrics)==6 and len(snapshots)==1200,
            "must preserve exactly 2x3 metrics and 6x200 terminal samples")
    require(info["output_counts"]=={
        "cfl_flux_metrics.csv":6,
        "snapshots.csv":1200,
        "step_budget.csv":len(ledger)},
        "CSV cardinalities inconsistent with run provenance")
    require([(r["method"],float(r["cfl_requested"])) for r in metrics]==EXPECTED_PAIRS,
            "incomplete, duplicate or disordered six CFL/flux keys")

    # Historical ET005-02 result is immutable and is a separate identity gate.
    official=Path(archive_root)/"ET005-02"
    for name,value in FROZEN_ET00502_HASHES.items():
        require(sha(official/name)==value,
                "previously frozen ET005-02 dataset changed: "+name)
    previous=read(official/"metrics.csv")
    historical=[r for r in previous if float(r["time"])==.5]
    require(len(historical)==1 and historical[0]["method"]=="Godunov" and
            abs(float(historical[0]["L1_cellavg"])-.003785807533217071)<1e-14,
            "ET005-02 scientific baseline not the registered original")
    original=original_means()
    initial_mass=DX*float(np.sum(original))
    reference=integral_exact()
    max_replay=max_balance=max_boundary=max_ref_error=0.0
    verified=[]
    total_steps=0
    for index,(method,cfl) in enumerate(EXPECTED_PAIRS):
        u=original.copy()
        t=0.0
        count=0
        dt_last=None
        inflow=0.0
        dt_seq=[]
        trace=[]
        ledger_group=[r for r in ledger if (r["method"],float(r["cfl_requested"]))==
                      (method,cfl)]
        metric=metrics[index]
        for target in TIMES:
            while target-t>2e-14:
                require(count<len(ledger_group),"missing time-step ledger rows")
                s=ledger_group[count]
                dt=min(cfl*DX,target-t)
                require(s["method"]==method and
                        float(s["cfl_requested"])==cfl and
                        int(s["step"])==count+1,
                        "step identity or method altered")
                near(s["time_start"],t,"step starting time",3e-14)
                near(s["dt"],dt,"registered step/cutback",3e-14)
                near(s["cfl_actual"],dt/DX,"actual CFL",3e-13)
                require(dt>0.0 and dt<=cfl*DX+3e-14 and dt/DX<=1.0,
                        "step not positive or CFL violation")
                faces=[scalar_flux(1.,float(u[0]),method)]
                for j in range(1,N):
                    faces.append(scalar_flux(float(u[j-1]),float(u[j]),method))
                faces.append(scalar_flux(float(u[-1]),0.,method))
                max_boundary=max(max_boundary,
                    abs(float(s["F_left"])-faces[0]),
                    abs(float(s["F_right"])-faces[-1]))
                near(s["F_left"],faces[0],"left exterior scalar flux")
                near(s["F_right"],faces[-1],"right exterior scalar flux")
                near(faces[0],.5,"registered left exterior trace flux",1e-10)
                near(faces[-1],0.,"registered right exterior trace flux",1e-10)
                m_before=DX*float(np.sum(u))
                new=np.array([u[j]-dt/DX*(faces[j+1]-faces[j])
                              for j in range(N)],dtype=np.float64)
                m_after=DX*float(np.sum(new))
                inc=dt*(faces[0]-faces[-1])
                inflow+=inc
                step_defect=(m_after-m_before)-inc
                mass_defect=(m_after-initial_mass)-inflow
                max_balance=max(max_balance,abs(step_defect),abs(mass_defect))
                near(s["mass_before"],m_before,"pre-step mass")
                near(s["mass_after"],m_after,"post-step mass")
                near(s["flux_budget_increment"],inc,"step net inflow")
                near(s["flux_budget_cumulative"],inflow,"cumulative net inflow")
                near(s["step_mass_defect"],step_defect,"step conservation defect")
                near(s["mass_defect"],mass_defect,"cumulative conservation defect")
                require(np.isfinite(new).all() and
                        new.min()>=-1e-12 and new.max()<=1.+1e-12,
                        "nonfinite or discrete maximum principle violated")
                near(s["min"],new.min(),"step minimum")
                near(s["max"],new.max(),"step maximum")
                tv=float(np.abs(np.diff(new)).sum()+abs(new[0]-1.)+abs(new[-1]))
                near(s["total_variation"],tv,"Riemann TV")
                t+=dt
                near(s["time_end"],t,"step ending time",3e-14)
                require(t<=target+3e-14,"stepped past registered checkpoint")
                u=new
                count+=1
                dt_last=dt
                dt_seq.append(dt)
            near(t,target,"missed registered checkpoint",3e-14)
            t=target
            trace.append(count)
        require(count==len(ledger_group) and len(dt_seq)==count,
                "missing/extra time-step ledger entries")
        total_steps+=count
        require(info["step_schedules"][f"{method}:CFL={cfl}"]=={
            "steps":count,
            "checkpoints_steps":{str(q):v for q,v in zip(TIMES,trace)},
            "dt_count":count},
            "time-step metadata/checkpoint scheduling mismatch")
        frame=snapshots[index*N:(index+1)*N]
        require(len(frame)==N and [int(r["i"]) for r in frame]==list(range(N)),
                "snapshot grid IDs missing or out of order")
        observed=np.array([float(r["U"]) for r in frame],dtype=np.float64)
        exact_from_output=np.array([float(r["U_exact"]) for r in frame],dtype=np.float64)
        max_replay=max(max_replay,float(np.max(abs(observed-u))))
        max_ref_error=max(max_ref_error,float(np.max(abs(exact_from_output-reference))))
        require(np.isfinite(observed).all() and
                max_replay<3e-12 and max_ref_error<3e-12,
                "numerical face replay or exact cell integral differs")
        for i,r in enumerate(frame):
            xl=-1.0+i*DX
            require(r["method"]==method and
                    float(r["cfl_requested"])==cfl and
                    int(r["N"])==N and float(r["time"])==.5,
                    "wrong snapshot method/time/data-source")
            near(r["x_left"],xl,"snapshot left cell face",2e-14)
            near(r["x_right"],xl+DX,"snapshot right cell face",2e-14)
            near(r["x_center"],xl+.5*DX,"snapshot center",2e-14)

        m=DX*float(np.sum(observed))
        exact_l1=DX*float(np.abs(observed-reference).sum())
        diag=shape_diagnostics(observed)
        require(metric["experiment_id"]=="ET005-05" and
                metric["case_id"]=="shock_1_0" and
                metric["method"]==method and metric["boundary"]=="riemann" and
                metric["TV_boundary"]=="riemann_fixed_exterior" and
                metric["reference_kind"]=="exact_shock_cellavg" and
                metric["status"]=="computed_unvalidated",
                "invalid scientific metric identity")
        near(metric["N"],N,"N",0)
        near(metric["dx"],DX,"dx",0)
        near(metric["cfl_requested"],cfl,"requested CFL",0)
        near(metric["time"],.5,"terminal time",0)
        require(metric["LF_alpha"]==("1.0" if method=="LF_alpha1" else ""),
                "LF alpha not globally fixed or Godunov assigned artificial alpha")
        require(int(metric["steps"])==count,"final step total wrong")
        near(metric["dt_last"],dt_last,"last actual time step",3e-14)
        near(metric["initial_mass"],initial_mass,"initial mass")
        near(metric["mass"],m,"observed mass")
        near(metric["exact_mass"],1.25,"exact mass")
        near(metric["net_boundary_flux_integral"],inflow,"cumulative mass inflow")
        near(metric["mass_defect"],m-initial_mass-inflow,"reported conservation defect")
        require(abs(m-1.25)<1e-11,"failed nonperiodic analytic mass budget")
        near(metric["min"],observed.min(),"min")
        near(metric["max"],observed.max(),"max")
        near(metric["total_variation"],diag["TV"],"total variation")
        near(metric["L1_cellavg"],exact_l1,"cellavg L1 independent integral",3e-12)
        near(metric["shock_speed_exact"],.5,"Rankine-Hugoniot speed",0)
        near(metric["shock_x_exact"],.25,"Rankine-Hugoniot terminal location",0)
        near(metric["shock_x_half"],diag["x_half"],"observed .5 crossing",3e-13)
        near(metric["shock_x_half_error"],diag["x_half"]-.25,"observed position error",3e-13)
        near(metric["shock_x_mass"],-1.+m,"mass inferred location",3e-12)
        near(metric["transition_low"],.1,"strict transition lower",0)
        near(metric["transition_high"],.9,"strict transition upper",0)
        require(int(metric["transition_cells"])==diag["transition_cells"]
                and int(metric["transition_segments"])==diag["transition_segments"],
                "transition cells/segments do not match strict 0.1<U<0.9")
        near(metric["transition_width"],diag["transition_width"],
             "transition region total geometric width",0)
        near(metric["quadratic_content"],diag["quadratic_content"],
             "integrated U^2/2 content",3e-12)
        verified.append({
            "method":method,"cfl_requested":cfl,
            "L1_cellavg":exact_l1,
            "shock_x_half":diag["x_half"],
            "shock_x_half_error":diag["x_half"]-.25,
            "transition_cells":diag["transition_cells"],
            "transition_segments":diag["transition_segments"],
            "transition_width":diag["transition_width"],
            "quadratic_content":diag["quadratic_content"],
            "mass":m,"mass_defect":m-initial_mass-inflow,
            "steps":count,"checkpoint_step_counts":trace
        })
    require(total_steps==len(ledger),"ledger has unregistered method/CFL rows")
    require(max_replay<3e-12 and max_balance<1e-11 and
            max_boundary<2e-12 and max_ref_error<3e-12,
            "hard independent full step verification failed")
    # Same CFL, same initialization, exactly the same physical time grid.
    for cfl in CFLS:
        def dt_list(method):
            group=[r for r in ledger
                   if r["method"]==method and float(r["cfl_requested"])==cfl]
            return [float(r["dt"]) for r in group]
        d1,d2=dt_list("Godunov"),dt_list("LF_alpha1")
        require(len(d1)==len(d2) and all(a==b for a,b in zip(d1,d2)),
                "comparison not on identical time-step schedule")
    return {
        "experiment_id":"ET005-05","status":"PASS",
        "source_git_sha":info["source_git_sha"],
        "config_sha256":info["config_sha256"],
        "design_sha256":info["design_sha256"],
        "independent_verifier":"scalar_201_face_per_step_plus_geometric_exact_cell_means",
        "verified_pairs":verified,"metric_rows":len(metrics),
        "snapshot_rows":len(snapshots),"step_budget_rows":len(ledger),
        "max_mass_budget_defect":max_balance,
        "max_boundary_flux_replay_error":max_boundary,
        "max_independent_FVM_state_error":max_replay,
        "max_independent_exact_mean_error":max_ref_error,
        "archived_ET00502_hashes":FROZEN_ET00502_HASHES,
        "output_sha256":info["output_sha256"]
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--design",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--archive-root",required=True,type=Path)
    args=parser.parse_args()
    checks=Path(args.out)/"checks.json"
    try:
        report=audit(args.out,args.config,args.design,args.archive_root)
    except Exception as ex:
        report={"experiment_id":"ET005-05","status":"FAIL",
                "failed_reason":f"{type(ex).__name__}: {ex}"}
        checks.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                          encoding="utf-8")
        print(report["failed_reason"],file=sys.stderr)
        return 1
    checks.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=="__main__":
    sys.exit(main())
