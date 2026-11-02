#!/usr/bin/env python3
"""Independent ET007-02 scientific verification; NEVER imports production code.

Checks the analytic affine solution T=x, the complete 45-degree tensor flux
(-5.5,-4.5), and the M0 omission of q_y. Reconstructs N=64 original face
data, both methods' internal Omega_h budget, boundary-value identity, and
the stored numerical diagnostics from raw CSV. N=32 is checked against the
independent analytic identities and scientific summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARTICLE = "tutorial-007"
EXPERIMENT = "ET007-02"
GRID = (32, 64)
METHODS = ("M0", "M1")
KXX, KXY, KYY = 5.5, 4.5, 5.5
TRUE_QX, TRUE_QY = -5.5, -4.5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def near(value: float, expected: float, title: str,
         atol: float = 1e-9, rtol: float = 0.0) -> None:
    require(math.isfinite(value) and math.isclose(value, expected,
                  rel_tol=rtol, abs_tol=atol),
            f"{title}: observed={value:.17g}, expected={expected:.17g}")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def independent_corner_test() -> dict:
    # N=4 nonhomogeneous Dirichlet T=xy, so f=-2Kxy=-9.
    n = 4
    h = 1/n
    def value(i: int, j: int) -> float:
        return (i*h)*(j*h)
    offsets = [(0,0,2*(KXX+KYY)/h**2),
               (1,0,-KXX/h**2),(-1,0,-KXX/h**2),
               (0,1,-KYY/h**2),(0,-1,-KYY/h**2),
               (1,1,-KXY/(2*h**2)),(-1,-1,-KXY/(2*h**2)),
               (1,-1,+KXY/(2*h**2)),(-1,1,+KXY/(2*h**2))]
    evidence = []
    for i, j in ((1,1),(1,3),(3,1),(3,3)):
        interior, rhs = 0.0, -9.0
        boundary_terms = 0
        for di, dj, coefficient in offsets:
            ii,jj = i+di,j+dj
            contribution = coefficient*value(ii,jj)
            if ii in (0,n) or jj in (0,n):
                rhs -= contribution
                if abs(contribution) > 0: boundary_terms += 1
            else:
                interior += contribution
        near(interior,rhs,f"nonzero Dirichlet corner ({i},{j})",atol=4e-13)
        evidence.append({"node":[i,j], "known_nonzero_terms":boundary_terms,
                         "discrete_row_residual":interior-rhs})
        if (i,j)==(3,3):
            require(boundary_terms==5,"missing diagonal Dirichlet edge/corner")
            near(rhs,123.,"nonzero upper-right corner",atol=4e-13)
    return {"status":"PASS","near_corner_rows":evidence}


def read_full_fields(root: Path, n: int, case: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    h=1.0/n
    path=root/"raw"/case
    temp=rows(path/"temperature.csv")
    vertical=rows(path/"flux_x.csv")
    horizontal=rows(path/"flux_y.csv")
    require(len(temp)==(n+1)**2,"incomplete full nodal temperature field")
    require(len(vertical)==n*(n-1) and len(horizontal)==n*(n-1),
            "incomplete transverse/normal face data")
    u=np.empty((n+1,n+1),dtype=float)
    qx=np.empty((n,n-1),dtype=float)
    qy=np.empty((n-1,n),dtype=float)
    # Frozen format: i major, then j; all positions checked independently.
    for record,(i,j) in zip(temp,((i,j) for i in range(n+1) for j in range(n+1)),strict=True):
        require((int(record["i"]),int(record["j"]))==(i,j),"unsorted nodal sample")
        near(float(record["x"]),i*h,"nodal x",atol=2e-14)
        near(float(record["y"]),j*h,"nodal y",atol=2e-14)
        u[i,j]=float(record["u"])
    for record,(i,j) in zip(vertical,((i,j) for i in range(n) for j in range(1,n)),strict=True):
        require((int(record["i"]),int(record["j"]))==(i,j),"unsorted vertical face")
        near(float(record["x"]),(i+0.5)*h,"vertical x",atol=2e-14)
        near(float(record["y"]),j*h,"vertical y",atol=2e-14)
        qx[i,j-1]=float(record["q_x"])
    for record,(i,j) in zip(horizontal,((i,j) for i in range(1,n) for j in range(n)),strict=True):
        require((int(record["i"]),int(record["j"]))==(i,j),"unsorted horizontal face")
        near(float(record["x"]),i*h,"horizontal x",atol=2e-14)
        near(float(record["y"]),(j+0.5)*h,"horizontal y",atol=2e-14)
        qy[i-1,j]=float(record["q_y"])
    require(np.all(np.isfinite(u)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qy)),
            "saved numerical data are not finite")
    return u,qx,qy


def audit_full_fields(root: Path, n: int, case: str, method: str,
                      metric: dict, config: dict) -> dict:
    h=1.0/n
    u,qx,qy=read_full_fields(root,n,case)
    analytic=np.arange(n+1,dtype=float)[:,None]*h+np.zeros((1,n+1))
    max_temp_error=float(np.max(np.abs(u-analytic)))
    require(max_temp_error<=config["tolerances"]["affine_solution_linf_max"],
            "N64 full nodal solution differs from exact T=x")
    # Independently reproduce BOTH face formulas, without importing production.
    expected_qx=-KXX*(u[1:,1:n]-u[:-1,1:n])/h
    expected_qy=-KYY*(u[1:n,1:]-u[1:n,:-1])/h
    if method=="M1":
        expected_qx-=(KXY/(4*h))*(u[:-1,2:]-u[:-1,:-2]+u[1:,2:]-u[1:,:-2])
        expected_qy-=(KXY/(4*h))*(u[2:,:-1]-u[:-2,:-1]+u[2:,1:]-u[:-2,1:])
    require(float(np.max(np.abs(qx-expected_qx)))<=2e-11,"saved qx is not the stated stencil flux")
    require(float(np.max(np.abs(qy-expected_qy)))<=2e-11,"saved qy is not the stated stencil flux")
    qx_err=float(np.max(np.abs(qx-TRUE_QX)))
    qy_err=float(np.max(np.abs(qy-TRUE_QY)))
    temperature_err=float(np.max(np.abs(u[1:n,1:n]-analytic[1:n,1:n])))
    temperature_l2=float(np.sqrt(h*h*np.sum((u[1:n,1:n]-analytic[1:n,1:n])**2)))
    near(float(metric["T_error_l2"]),temperature_l2,"reported full-field T L2",atol=2e-11)
    near(float(metric["T_error_linf"]),temperature_err,"reported full-field T Linf",atol=2e-11)
    near(float(metric["q_x_error_linf"]),qx_err,"reported full-field qx Linf",atol=2e-11)
    near(float(metric["q_y_error_linf"]),qy_err,"reported full-field qy Linf",atol=2e-11)
    near(float(metric["q_x_mean"]),float(np.mean(qx)),"reported true qx mean",atol=2e-11)
    near(float(metric["q_y_mean"]),float(np.mean(qy)),"reported true qy mean",atol=2e-11)
    scale=max(1.0,float(np.max(np.abs(h*qx))),float(np.max(np.abs(h*qy))))
    # The source is zero, and Omega_h contains only interior vertex-dual cells.
    balances=h*(qx[1:,:]-qx[:-1,:]+qy[:,1:]-qy[:,:-1])
    balance_linf=float(np.max(np.abs(balances)))
    budget=h*(np.sum(qx[-1,:])-np.sum(qx[0,:])+np.sum(qy[:,-1])-np.sum(qy[:,0]))
    near(float(metric["dual_cell_balance_linf"]),balance_linf,"local dual budget",atol=3e-11)
    near(float(metric["dual_boundary_budget_defect"]),float(budget),
         "whole dual Omega_h budget",atol=3e-11)
    require(balance_linf/scale<=config["tolerances"]["dual_cell_balance_scaled_max"],
            "dual-cell local conservation criterion exceeded")
    require(abs(float(budget))/scale<=config["tolerances"]["dual_cell_balance_scaled_max"],
            "dual Omega_h budget criterion exceeded")
    if method=="M0":
        require(float(np.max(np.abs(qy)))<=config["tolerances"]["affine_true_m1_flux_linf_max"],
                "omitting cross conductivity did not produce near-zero numerical qy")
        near(qy_err,4.5,"M0 physical transverse-flux error")
    else:
        require(float(np.max(np.abs(qy-TRUE_QY)))
                <=config["tolerances"]["affine_true_m1_flux_linf_max"],
                "M1 complete physical transverse flux is wrong")
    return {"case_id":case,"full_nodes_checked":len(u.ravel()),
            "face_values_checked":qx.size+qy.size,
            "actual_mean_qx":float(np.mean(qx)),
            "actual_mean_qy":float(np.mean(qy)),
            "actual_qy_physical_error_linf":qy_err,
            "max_temperature_error":max_temp_error,
            "max_dual_cell_balance_defect":balance_linf,
            "Omega_h_boundary_budget_defect":float(budget)}


def verify(config_path: Path, result: Path) -> dict:
    spec=json.loads(config_path.read_text(encoding="utf-8"))
    require(spec["article_id"]==ARTICLE and spec["design_version"]=="v0.2"
            and spec["status"]=="science_parameters_frozen","design is not frozen")
    stage=next(s for s in spec["experiment_sets"] if s["id"]==EXPERIMENT)
    require(stage["theta_deg"]==45.0 and stage["field"]=="affine"
            and stage["grids_N"]==[32,64] and stage["methods"]==["M0","M1"]
            and stage["save_full_fields_N"]==[64], "ET007-02 stage was modified")
    require(spec["tensor_principal"]==[10.,1.],"physical tensor changed")
    run=json.loads((result/"run.json").read_text(encoding="utf-8"))
    require(run["article_id"]==ARTICLE and run["experiment"]==EXPERIMENT
            and run["status"]=="completed","not a successful ET007-02 run")
    require(run["executed_case_count"]==4 and run["planned_full_case_count"]==28
            and run["complete_28_case_program"] is False,
            "must not declare ET007-03/04 already computed")
    require(run["config_sha256"]==digest(config_path),"frozen configuration identity differs")
    source_root=Path(__file__).resolve().parents[1]
    for path,sha in run["source_sha256"].items():
        require(digest(source_root/path)==sha,"source provenance mismatch: "+path)
    for path,sha in run["scientific_sha256"].items():
        require(digest(result/path)==sha,"science data identity mismatch: "+path)
    summary=rows(result/"summary.csv")
    expected=[f"affine-p45-N{n}-{method}" for n in GRID for method in METHODS]
    require([r["case_id"] for r in summary]==expected
            and run["run_case_ids"]==expected, "four unique cases or ordering altered")
    real_science=sorted(p.relative_to(result).as_posix() for p in result.rglob("*.csv"))
    require(len(real_science)==7,"only 7 science CSV expected: stage summary plus six N64 fields")
    observed=[]
    for case,record in zip(expected,summary,strict=True):
        n=int(record["N"]);method=record["method"]
        require(record["field"]=="affine" and record["experiment"]==EXPERIMENT
                and int(float(record["theta_deg"]))==45
                and record["solver_status"]=="solved", "invalid actual solver case")
        m=n-1
        require(int(record["n_unknowns"])==m*m,"wrong interior system size")
        nnz=5*m*m-4*m if method=="M0" else 9*m*m-12*m+4
        require(int(record["nnz"])==nnz,"unexpected number of matrix nonzeros")
        for k,v in [("K_xx",KXX),("K_xy",KXY),("K_yy",KYY),
                    ("q_x_true",TRUE_QX),("q_y_true",TRUE_QY)]:
            near(float(record[k]),v,case+"/"+k,atol=3e-13)
        require(float(record["linear_residual_linf_relative"])
                <=spec["tolerances"]["linear_residual_linf_relative_max"],
                "actual sparse solver residual failed")
        require(float(record["T_error_linf"])
                <=spec["tolerances"]["affine_solution_linf_max"],
                "a temperature solver is not exact for affine Dirichlet")
        require(float(record["q_x_error_linf"])
                <=spec["tolerances"]["affine_true_m1_flux_linf_max"],
                "real physical qx incorrectly recovered")
        for key in ("dual_cell_balance_scaled","dual_boundary_budget_scaled"):
            require(float(record[key])<=spec["tolerances"]["dual_cell_balance_scaled_max"],
                    "dual-cell discrete balance failed: "+case)
        near(float(record["q_x_mean"]),TRUE_QX,case+"/mean qx",atol=1e-9)
        if method=="M0":
            near(float(record["q_y_mean"]),0.,case+"/omitted cross flux",atol=1e-9)
            near(float(record["q_y_error_linf"]),4.5,case+"/real flux loss",atol=1e-9)
        else:
            near(float(record["q_y_mean"]),TRUE_QY,case+"/correct full tensor flux",atol=1e-9)
            require(float(record["q_y_error_linf"])
                    <=spec["tolerances"]["affine_true_m1_flux_linf_max"],
                    "full tensor numerical flux deviates from physical flux")
        rawcase=json.loads((result/"raw"/case/"run.json").read_text(encoding="utf-8"))
        require(rawcase["case_id"]==case and rawcase["status"]=="solved"
                and rawcase["configuration_sha256"]==digest(config_path),
                "incorrect per-case provenance")
        require(rawcase["source_sha256"]==run["source_sha256"],
                "per-case production source identity differs")
        for name,sha in rawcase["scientific_csv_sha256"].items():
            require(digest(result/"raw"/case/name)==sha,
                    "saved independent scientific field digest changed")
        if n==64:
            require(set(rawcase["scientific_csv_sha256"])
                    =={"temperature.csv","flux_x.csv","flux_y.csv"},
                    "missing the preplanned N64 full-field evidence")
            observed.append(audit_full_fields(result,n,case,method,record,spec))
        else:
            require(not rawcase["scientific_csv_sha256"],"unplanned N32 full fields")
    require(len(observed)==2,"both N64 methods must be checked from genuine CSV")
    # Check sign reversal analytically without counting an additional PDE solve.
    sign_reversal={"theta_deg":-45,"analytic_q_y":4.5}
    corners=independent_corner_test()
    return {"status":"PASS","article_id":ARTICLE,"experiment":EXPERIMENT,
            "run_source_commit":run.get("source_commit"),
            "independent_production_import":False,"actual_sparse_case_count":4,
            "planned_total_case_count":28,"scientific_csv_count":len(real_science),
            "true_physical_flux":{"q_x":TRUE_QX,"q_y":TRUE_QY},
            "M0_omits_q_y":True,"M1_recovers_q_y":True,
            "N64_raw_field_checks":observed,"analytic_sign_reversal":sign_reversal,
            "nonzero_Dirichlet_corner_regression":corners,
            "scope":"Omega_h_dual_conservation_not_physical_boundary_validation"}


def main() -> None:
    args=argparse.ArgumentParser()
    args.add_argument("--config",required=True,type=Path)
    args.add_argument("--results",required=True,type=Path)
    args.add_argument("--out",required=True,type=Path)
    opt=args.parse_args()
    if opt.out.exists():
        raise FileExistsError(f"refusing to overwrite: {opt.out}")
    report=verify(opt.config.resolve(),opt.results.resolve())
    opt.out.parent.mkdir(parents=True,exist_ok=True)
    opt.out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,
                                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("INDEPENDENT_ET007_02_PASS",report["actual_sparse_case_count"],
          "cases",len(report["N64_raw_field_checks"]),
          "full face-field audits",flush=True)


if __name__=="__main__":
    main()
