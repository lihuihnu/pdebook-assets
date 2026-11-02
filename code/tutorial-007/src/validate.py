#!/usr/bin/env python3
"""Independent ET007-01 checker. Does NOT import the production model or solver.

Scientific independent benchmark: the EXACT solution of the assembled
aligned-grid discrete system is A_N sin(pi*i*h) sin(pi*j*h).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(observed: float, expected: float, name: str,
          *, rel: float = 1e-9, abs_tol: float = 3e-11) -> None:
    check(math.isfinite(observed) and math.isfinite(expected) and
          math.isclose(observed, expected, rel_tol=rel, abs_tol=abs_tol),
          f"{name}: observed {observed:.17g}, expected {expected:.17g}")


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def discrete_amplitude(n: int) -> float:
    h = 1.0 / n
    return (math.pi*h / (2*math.sin(math.pi*h/2)))**2


def exact_metric(n: int, kxx: float, kyy: float) -> dict[str,float]:
    """Independent analytic evaluation of the aligned-grid discrete mode."""
    h = 1.0/n
    a = discrete_amplitude(n)
    x = np.arange(n+1)*h
    sx = np.sin(math.pi*x)
    interior = (a-1.)*sx[1:n,None]*sx[None,1:n]
    # Flux of discrete sine mode, compared at each *face centre* to physical q.
    xf = (np.arange(n)+0.5)*h
    yf = np.arange(1,n)*h
    delta = a*(2.*math.sin(math.pi*h/2)/h)-math.pi
    qx_error = abs(kxx*delta)*np.abs(np.cos(math.pi*xf))[:,None]*np.abs(np.sin(math.pi*yf))[None,:]
    qy_error = abs(kyy*delta)*np.abs(np.sin(math.pi*yf))[:,None]*np.abs(np.cos(math.pi*xf))[None,:]
    return {
        "T_error_l2":float(math.sqrt(h*h*float(np.sum(interior*interior)))),
        "T_error_linf":float(np.max(np.abs(interior))),
        "q_x_error_linf":float(np.max(qx_error)),
        "q_y_error_linf":float(np.max(qy_error)),
    }


def audit_saved_field(raw: Path, n: int, metric: dict[str,str],
                      tolerances: dict) -> dict:
    h = 1.0/n
    temp = read_csv(raw/"temperature.csv")
    vertical = read_csv(raw/"flux_x.csv")
    horizontal = read_csv(raw/"flux_y.csv")
    check(len(temp)==(n+1)**2, "missing nodal temperatures")
    check(len(vertical)==n*(n-1) and len(horizontal)==n*(n-1),
          "missing aligned dual faces")
    u = np.empty((n+1,n+1),dtype=float)
    qx = np.empty((n,n-1),dtype=float)
    qy = np.empty((n-1,n),dtype=float)
    for entry, i in zip(temp, ((i,j) for i in range(n+1) for j in range(n+1)), strict=True):
        ii,jj=i
        check((int(entry["i"]),int(entry["j"]))==(ii,jj),"nodal CSV out of order")
        close(float(entry["x"]),ii*h,"node x coordinate",abs_tol=3e-15)
        close(float(entry["y"]),jj*h,"node y coordinate",abs_tol=3e-15)
        u[ii,jj]=float(entry["u"])
    for entry, pos in zip(vertical, ((i,j) for i in range(n) for j in range(1,n)), strict=True):
        i,j=pos
        check((int(entry["i"]),int(entry["j"]))==pos,"vertical faces unsorted")
        close(float(entry["x"]),(i+0.5)*h,"face x coordinate",abs_tol=3e-15)
        close(float(entry["y"]),j*h,"face y coordinate",abs_tol=3e-15)
        qx[i,j-1]=float(entry["q_x"])
    for entry, pos in zip(horizontal, ((i,j) for i in range(1,n) for j in range(n)), strict=True):
        i,j=pos
        check((int(entry["i"]),int(entry["j"]))==pos,"horizontal faces unsorted")
        close(float(entry["x"]),i*h,"face x coordinate",abs_tol=3e-15)
        close(float(entry["y"]),(j+0.5)*h,"face y coordinate",abs_tol=3e-15)
        qy[i-1,j]=float(entry["q_y"])
    check(np.all(np.isfinite(u)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qy)),
          "nonfinite full field")
    a = discrete_amplitude(n)
    grid = np.sin(math.pi*np.arange(n+1)*h)
    ref = a*grid[:,None]*grid[None,:]
    ref[[0,n],:]=0.
    ref[:,[0,n]]=0.
    maxerr=float(np.max(np.abs(u-ref)))
    check(maxerr<=tolerances["analytic_discrete_solution_linf_max"],
          f"analytic discrete sine mode mismatch: {maxerr}")
    # Independent face difference formulas, not a production import:
    qx_expected=-10.*(u[1:,1:n]-u[:-1,1:n])/h
    qy_expected=-(u[1:n,1:]-u[1:n,:-1])/h
    check(float(np.max(np.abs(qx-qx_expected)))<1e-9,"vertical face CSV inconsistent with nodal field")
    check(float(np.max(np.abs(qy-qy_expected)))<1e-9,"horizontal face CSV inconsistent with nodal field")
    # Every internal face is indexed exactly once; adjacent cells use opposite sign.
    source=11.*math.pi**2*grid[1:n,None]*grid[None,1:n]
    balance=h*(qx[1:]-qx[:-1]+qy[:,1:]-qy[:,:-1])-h*h*source
    scale=max(1.,float(np.max(np.abs(h*qx))),
              float(np.max(np.abs(h*qy))),h*h*float(np.max(np.abs(source))))
    interior_defect=float(np.max(np.abs(balance)))
    budget=h*(np.sum(qx[-1,:])-np.sum(qx[0,:])+
              np.sum(qy[:,-1])-np.sum(qy[:,0]))-h*h*np.sum(source)
    close(float(metric["dual_cell_balance_linf"]),interior_defect,
          "saved-field local conservation",rel=2e-3,abs_tol=2e-10)
    check(interior_defect/scale<=tolerances["dual_cell_balance_scaled_max"],
          "local dual balance exceeds frozen criterion")
    check(abs(float(budget))/scale<=tolerances["dual_cell_balance_scaled_max"],
          "dual-area global budget exceeds frozen criterion")
    return {"N":n,"nodal_max_discrete_solution_error":maxerr,
            "local_balance_linf":interior_defect,
            "global_dual_budget_absolute":abs(float(budget)),
            "face_count":len(vertical)+len(horizontal)}


def verify(config: Path, result: Path) -> dict:
    spec=json.loads(config.read_text(encoding="utf-8"))
    record=json.loads((result/"run.json").read_text(encoding="utf-8"))
    check(spec["status"]=="science_parameters_frozen" and spec["design_version"]=="v0.2",
          "science design not frozen")
    check(record["article_id"]=="tutorial-007" and
          record["experiment"]=="ET007-01" and record["status"]=="completed",
          "wrong or incomplete experiment")
    check(not record["complete_28_case_program"] and record["executed_case_count"]==8
          and record["planned_full_case_count"]==28,"future cases must not be marked executed")
    check(sha(config)==record["config_sha256"],"config/source provenance differs")
    for source, fingerprint in record["source_sha256"].items():
        check(sha(Path(__file__).resolve().parent.parent/source)==fingerprint,
              "production source changed: "+source)
    for name, digest in record["artifact_sha256"].items():
        check(sha(result/name)==digest,"scientific artifact modified: "+name)
    observations=read_csv(result/"summary.csv")
    check(len(observations)==8,"expected exactly eight results")
    ordered=[f"sinesine-p00-N{n}-{method}"
             for n in (16,32,64,128) for method in ("M0","M1")]
    check([m["case_id"] for m in observations]==ordered,"incomplete or unsorted stage")
    checkpoints=[]
    t=spec["tolerances"]
    for m in observations:
        case=m["case_id"]
        n=int(m["N"])
        method=m["method"]
        check(m["solver_status"]=="solved" and m["field"]=="sinesine"
              and float(m["theta_deg"])==0.0,"invalid solver status/method")
        check(int(m["n_unknowns"])==(n-1)**2,"incorrect system dimension")
        check(int(m["nnz"])==5*(n-1)**2-4*(n-1),"not the aligned five-point matrix")
        for key,value in {"K_xx":10.0,"K_xy":0.0,"K_yy":1.0}.items():
            close(float(m[key]),value,"tensor "+key,abs_tol=1e-13)
        check(float(m["linear_residual_linf_relative"])
              <=t["linear_residual_linf_relative_max"],"linear residual gate failed")
        for key in ("dual_cell_balance_scaled","dual_boundary_budget_scaled"):
            check(float(m[key])<=t["dual_cell_balance_scaled_max"],
                  "discrete dual conservation failed: "+case+"/"+key)
        expected=exact_metric(n,10.,1.)
        for metric,value in expected.items():
            close(float(m[metric]),value,case+"/"+metric,rel=2e-7,abs_tol=1e-9)
        info=json.loads((result/"raw"/case/"run.json").read_text(encoding="utf-8"))
        check(info["case_id"]==case and info["status"]=="solved" and
              info["config_sha256"]==sha(config),"case provenance altered")
        for filename,hashed in info["scientific_csv_sha256"].items():
            check(sha(result/"raw"/case/filename)==hashed,
                  "raw field file altered")
        if n==64:
            check(set(info["scientific_csv_sha256"])==
                  {"temperature.csv","flux_x.csv","flux_y.csv"},"missing checkpoint arrays")
            checkpoints.append(audit_saved_field(result/"raw"/case,n,m,t))
        else:
            check(not info["scientific_csv_sha256"],"unexpected extra field exports")
    pairs=[]
    for n in (16,32,64,128):
        left,right=[m for m in observations if int(m["N"])==n]
        for key in ("T_error_l2","T_error_linf",
                    "q_x_error_linf","q_y_error_linf"):
            close(float(left[key]),float(right[key]),f"M0/M1 agreed {n} {key}",
                  rel=1e-9,abs_tol=1e-9)
        pairs.append({"N":n,"methods_analytically_equivalent":True})
    return {"status":"PASS","article_id":"tutorial-007","stage":"ET007-01",
            "config_sha256":sha(config),"run_source_commit":record.get("source_commit"),
            "actual_cases":8,"expected_cases":28,"science_csv_count":len(list(result.rglob("*.csv"))),
            "separately_derived_discrete_eigenmode":"A_N=(pi*h/(2*sin(pi*h/2)))^2",
            "all_metrics_checked_against_exact_discrete_mode":True,
            "saved_field_checks":checkpoints,"aligned_pairs":pairs,
            "production_solver_imported":False}


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",type=Path,required=True)
    arg.add_argument("--results",type=Path,required=True)
    arg.add_argument("--out",type=Path,required=True)
    parsed=arg.parse_args()
    if parsed.out.exists():
        raise FileExistsError(parsed.out)
    verdict=verify(parsed.config.resolve(),parsed.results.resolve())
    parsed.out.parent.mkdir(parents=True,exist_ok=True)
    parsed.out.write_text(json.dumps(verdict,indent=2,sort_keys=True,allow_nan=False)+"\n",
                          encoding="utf-8")
    print("INDEPENDENT_ET007_01_PASS",verdict["actual_cases"],
          "cases",len(verdict["saved_field_checks"]),"full-field checks",flush=True)


if __name__=="__main__":
    main()
