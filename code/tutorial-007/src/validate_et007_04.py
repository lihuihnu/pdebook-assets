#!/usr/bin/env python3
"""ET007-04 independent angle-sweep checker; never imports production solver.

M0: direct DST-I diagonalization of the five-point Dirichlet operator.
M1: independently written nine-point matrix-free operator solved with CG.
All 8 NEW runs and all 16 angle-comparison rows are checked. Older
0/45-degree scientific rows must match their pinned CSVs verbatim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import LinearOperator, cg


STAGE="ET007-04"
ANGLES=(0,15,30,45)
NEW_ANGLES=(15,30)
N_LIST=(64,128)
METHODS=("M0","M1")
SOURCE_HASHES={
    "ET007-01":"1788ec0ae48ead3903d6d09cb1bf8b999aa31b42d5f37728061fb2eefbd88233",
    "ET007-02":"bec0f2916cfecdab732eb4d48d81c8a9cd814353e337db7ea4384dad41b543d8",
    "ET007-03":"df816b6eb9ff4de8c9d1e8d182a1279bb3926448dbb399f81f9d1d53d44d9fa2",
}
METRIC_FIELDS=("T_error_l2","T_error_linf","q_x_error_linf",
               "q_y_error_linf","dual_cell_balance_linf",
               "dual_cell_balance_scaled","dual_boundary_budget_defect",
               "dual_boundary_budget_scaled")
PDE_ROW_FIELDS=("case_id","experiment","field","theta_deg","N","h",
                "method","K_xx","K_xy","K_yy","n_unknowns","nnz",
                "linear_residual_linf_relative","T_error_l2","T_error_linf",
                "q_x_error_linf","q_y_error_linf","dual_cell_balance_linf",
                "dual_cell_balance_scaled","dual_boundary_budget_defect",
                "dual_boundary_budget_scaled","solver_status")


def insist(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def approx(actual: float, wanted: float, context: str, scale: float) -> None:
    insist(math.isfinite(actual) and math.isfinite(wanted)
           and abs(actual-wanted)<=scale*max(1.,abs(wanted)),
           f"{context}: actual {actual:.17g} expected {wanted:.17g} error {abs(actual-wanted):.4g}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def tensor(angle_deg: int) -> tuple[float,float,float]:
    """Trigonometric expansion independent of production tensor()."""
    a=math.radians(angle_deg)
    c,s=math.cos(a),math.sin(a)
    return (1.+9*c*c,9*c*s,1.+9*s*s)


def analytic(n: int, angle_deg: int) -> tuple[np.ndarray,np.ndarray]:
    # Full physical tensor, NOT the M0 "corrected" source.
    h=1./n
    v=np.arange(1,n)*h
    x,y=np.meshgrid(v,v,indexing="ij")
    truth=np.sin(math.pi*x)*np.sin(math.pi*y)
    kxx,kxy,kyy=tensor(angle_deg)
    source=math.pi**2*((kxx+kyy)*truth
           -2*kxy*np.cos(math.pi*x)*np.cos(math.pi*y))
    return truth,source


def action(inside_vector: np.ndarray,n: int,
           angle: int,method: str) -> np.ndarray:
    h=1./n
    kxx,kxy,kyy=tensor(angle)
    u=np.zeros((n+1,n+1),dtype=np.float64)
    u[1:n,1:n]=np.asarray(inside_vector).reshape((n-1,n-1))
    m=u[1:n,1:n]
    out=-(kxx*(u[2:,1:n]-2*m+u[:-2,1:n])
         +kyy*(u[1:n,2:]-2*m+u[1:n,:-2]))/(h*h)
    if method=="M1":
        out-=kxy*(u[2:,2:]-u[2:,:-2]
                  -u[:-2,2:]+u[:-2,:-2])/(2*h*h)
    elif method!="M0":
        raise ValueError("unsupported checker method")
    return out.ravel()


def reference_solve(n: int, angle: int,
                    method: str, physical_source: np.ndarray
                    ) -> tuple[np.ndarray,dict]:
    kxx,kxy,kyy=tensor(angle)
    if method=="M0":
        j=np.arange(1,n)
        diagonal=4*n*n*(kxx*np.sin(math.pi*j/(2*n))[:,None]**2
                      +kyy*np.sin(math.pi*j/(2*n))[None,:]**2)
        uh=dstn(physical_source,type=1,norm="ortho")
        inside=idstn(uh/diagonal,type=1,norm="ortho")
        method_name="independent 2D DST-I diagonalization"
        iterations=0
    else:
        unknowns=(n-1)**2
        operator=LinearOperator((unknowns,unknowns),dtype=np.float64,
                                matvec=lambda v:action(v,n,angle,"M1"))
        inv_diagonal=1./(2*n*n*(kxx+kyy))
        pre=LinearOperator((unknowns,unknowns),dtype=np.float64,
                           matvec=lambda v:inv_diagonal*np.asarray(v))
        step=[0]
        def callback(_):
            step[0]+=1
        numerical,info=cg(operator,physical_source.ravel(),M=pre,
                          rtol=1e-13,atol=0,maxiter=4500,callback=callback)
        insist(info==0 and np.all(np.isfinite(numerical)),
               f"matrix-free CG unable to solve N={n} angle={angle}, info={info}")
        inside=numerical.reshape((n-1,n-1))
        method_name="independent matrix-free 9-point CG"
        iterations=step[0]
    residual=(action(inside.ravel(),n,angle,method)
              -physical_source.ravel())
    rel=float(np.max(np.abs(residual))/
              max(1.,float(np.max(np.abs(physical_source)))))
    insist(rel<=1e-10,
           f"independent reference solver residual above frozen gate: {rel}")
    return inside,{"reference_method":method_name,
                   "iterations":iterations,
                   "residual_linf_relative":rel}


def face_metrics(inside: np.ndarray,n: int,
                 angle: int,method: str) -> dict[str,float]:
    h=1./n
    kxx,kxy,kyy=tensor(angle)
    u=np.zeros((n+1,n+1),dtype=np.float64)
    u[1:n,1:n]=inside
    truth,source=analytic(n,angle)
    diff=inside-truth
    # Own face formula rather than importing production flux code.
    qx=-kxx*(u[1:,1:n]-u[:-1,1:n])/h
    qy=-kyy*(u[1:n,1:]-u[1:n,:-1])/h
    if method=="M1":
        uxdy=(u[:-1,2:]-u[:-1,:-2]+u[1:,2:]-u[1:,:-2])/(4*h)
        uydx=(u[2:,:-1]-u[:-2,:-1]+u[2:,1:]-u[:-2,1:])/(4*h)
        qx=qx-kxy*uxdy
        qy=qy-kxy*uydx
    xface=(np.arange(n)+0.5)*h
    ycenter=np.arange(1,n)*h
    gx=np.pi*np.cos(np.pi*xface)[:,None]*np.sin(np.pi*ycenter)[None,:]
    gy=np.pi*np.sin(np.pi*xface)[:,None]*np.cos(np.pi*ycenter)[None,:]
    exact_qx=-kxx*gx-kxy*gy
    gx=np.pi*np.cos(np.pi*ycenter)[:,None]*np.sin(np.pi*xface)[None,:]
    gy=np.pi*np.sin(np.pi*ycenter)[:,None]*np.cos(np.pi*xface)[None,:]
    # The x-direction on horizontal faces is ycenter, the y direction xface.
    exact_qy=-kxy*gx-kyy*gy
    net=h*(qx[1:]-qx[:-1]+qy[:,1:]-qy[:,:-1])-h*h*source
    scale=max(1.,float(np.max(np.abs(h*qx))),
              float(np.max(np.abs(h*qy))),
              h*h*float(np.max(np.abs(source))))
    budget=float(h*(np.sum(qx[-1])-np.sum(qx[0])
                    +np.sum(qy[:,-1])-np.sum(qy[:,0]))
                 -h*h*np.sum(source))
    return {
        "T_error_l2":float(np.sqrt(h*h*np.sum(diff**2))),
        "T_error_linf":float(np.max(np.abs(diff))),
        "q_x_error_linf":float(np.max(np.abs(qx-exact_qx))),
        "q_y_error_linf":float(np.max(np.abs(qy-exact_qy))),
        "dual_cell_balance_linf":float(np.max(np.abs(net))),
        "dual_cell_balance_scaled":float(np.max(np.abs(net))/scale),
        "dual_boundary_budget_defect":budget,
        "dual_boundary_budget_scaled":abs(budget)/scale,
    }


def compare_angular_view(view: list[dict],
                         new: list[dict],
                         old_0: list[dict],
                         old_45: list[dict],
                         run: dict) -> tuple[list[dict],list[dict]]:
    by_stage={"ET007-01":{r["case_id"]:r for r in old_0},
              "ET007-03":{r["case_id"]:r for r in old_45},
              "ET007-04":{r["case_id"]:r for r in new}}
    new_hash=sha256_cache=run["stage_summary_sha256"]
    provenance=[]
    for angle in ANGLES:
        for n in N_LIST:
            for method in METHODS:
                key=f"sinesine-p{angle:02d}-N{n}-{method}"
                index=len(provenance)
                row=view[index]
                stage=("ET007-01" if angle==0
                       else "ET007-03" if angle==45
                       else "ET007-04")
                insist(row["case_id"]==key and row["source_stage"]==stage
                       and row["experiment"]==stage
                       and float(row["theta_deg"])==angle
                       and int(row["N"])==n
                       and row["method"]==method,
                       "angular case order and frozen origin mismatched")
                expected=by_stage[stage].get(key)
                insist(expected is not None,"angle data has no real source "+key)
                for field in PDE_ROW_FIELDS:
                    insist(row[field]==expected[field],
                           f"{key}: {field} was altered in the reusable angular view")
                sourcehash=(new_hash if stage=="ET007-04"
                            else SOURCE_HASHES[stage])
                insist(row["source_summary_sha256"]==sourcehash,
                       f"{key}: source CSV SHA does not match")
                provenance.append({"case_id":key,"origin_stage":stage})
    insist(len(view)==16 and len(provenance)==16,"wrong angle comparison cardinality")
    comparisons=[]
    for angle in ANGLES:
        row_group=[x for x in view if int(float(x["theta_deg"]))==angle]
        for n in N_LIST:
            mm={r["method"]:r for r in row_group if int(r["N"])==n}
            insist(set(mm)=={"M0","M1"},"missing pair at this angle")
            per_case={"theta_deg":angle,"N":n,
                      "M0_temperature_L2":float(mm["M0"]["T_error_l2"]),
                      "M1_temperature_L2":float(mm["M1"]["T_error_l2"]),
                      "M0_flux_x_error_linf":float(mm["M0"]["q_x_error_linf"]),
                      "M1_flux_x_error_linf":float(mm["M1"]["q_x_error_linf"])}
            comparisons.append(per_case)
    return provenance,comparisons


def verify(config_path: Path,
           result: Path, et01: Path, et02: Path, et03: Path) -> dict:
    configuration=json.loads(config_path.read_text(encoding="utf-8"))
    insist(configuration["status"]=="science_parameters_frozen"
           and configuration["design_version"]=="v0.2"
           and configuration["tensor_principal"]==[10.,1.]
           and configuration["planned_unique_solve_count"]==28,
           "unfrozen or modified scientific design")
    stage=next(s for s in configuration["experiment_sets"]
               if s["id"]==STAGE)
    insist(stage["theta_deg_values"]==list(ANGLES)
           and stage["new_theta_deg_values"]==list(NEW_ANGLES)
           and stage["grids_N"]==list(N_LIST)
           and stage["methods"]==list(METHODS)
           and stage["reuse_case_ids_from"]==["ET007-01","ET007-03"]
           and stage["save_full_fields_N"]==[],
           "ET007-04 proposed case set changed")
    frozen_paths={"ET007-01":et01,"ET007-02":et02,"ET007-03":et03}
    previous={}
    for name,path in frozen_paths.items():
        insist(sha256(path)==SOURCE_HASHES[name],
               "historical frozen science changed: "+name)
        previous[name]=rows(path)
        insist(len(previous[name])=={"ET007-01":8,"ET007-02":4,"ET007-03":8}[name],
               "historical science lost cases: "+name)
        insist(len({r["case_id"] for r in previous[name]})==len(previous[name]),
               "duplicate case in "+name)
        insist(all(r["experiment"]==name and r["solver_status"]=="solved"
                   for r in previous[name]),"unaccepted historical data: "+name)
    run=json.loads((result/"run.json").read_text(encoding="utf-8"))
    insist(run["article_id"]=="tutorial-007" and run["experiment"]==STAGE
           and run["status"]=="completed"
           and run["executed_case_count"]==8
           and run["reused_case_count"]==8
           and run["earlier_accepted_case_count"]==20
           and run["angle_comparison_count"]==16
           and run["planned_full_case_count"]==28
           and not run["complete_28_case_program"]
           and not run["raw_fields_saved"],"wrong experiment stage identity")
    insist(run["config_sha256"]==sha256(config_path),"frozen config changed")
    insist(run["frozen_summary_sha256"]==SOURCE_HASHES,"reuse fingerprints differ")
    insist(run["stage_summary_sha256"]==sha256(result/"summary.csv"),
           "fresh summary fingerprint wrong")
    code_root=Path(__file__).resolve().parents[1]
    for file,checksum in run["source_sha256"].items():
        insist(sha256(code_root/file)==checksum,
               "production input source SHA differs: "+file)
    for file,checksum in run["scientific_sha256"].items():
        insist(sha256(result/file)==checksum,
               "archived science was modified: "+file)
    fresh=rows(result/"summary.csv")
    require_keys=[f"sinesine-p{angle:02d}-N{n}-{method}"
                  for angle in NEW_ANGLES for n in N_LIST for method in METHODS]
    insist(len(fresh)==8 and [x["case_id"] for x in fresh]==require_keys
           and run["new_case_ids"]==require_keys,
           "not exactly eight new cases")
    view=rows(result/"angle_comparison.csv")
    provenance,pairs=compare_angular_view(view,fresh,
                        previous["ET007-01"],previous["ET007-03"],run)
    insist(run["reused_case_ids"]==
           [p["case_id"] for p in provenance if p["origin_stage"]!=STAGE],
           "reused case registry differs from actual frozen science")
    float_tol=float(configuration["tolerances"]["independent_check_linf_relative_max"])
    insist(float_tol==1e-9,"scientific independent tolerance was edited")
    independent=[]
    for row in fresh:
        case=row["case_id"]
        n=int(row["N"])
        method=row["method"]
        angle=int(float(row["theta_deg"]))
        insist(method in METHODS and angle in NEW_ANGLES
               and int(row["n_unknowns"])==(n-1)**2
               and row["field"]=="sinesine"
               and row["solver_status"]=="solved"
               and row["experiment"]==STAGE,
               f"{case}: incorrect true production case")
        m=n-1
        nnz=5*m*m-4*m if method=="M0" else 9*m*m-12*m+4
        insist(int(row["nnz"])==nnz,f"{case}: incorrect sparse stencil")
        k=tensor(angle)
        for field,wanted in zip(("K_xx","K_xy","K_yy"),k):
            approx(float(row[field]),wanted,case+"/"+field,1e-13)
        insist(float(row["linear_residual_linf_relative"])
               <=configuration["tolerances"]["linear_residual_linf_relative_max"],
               f"{case}: production linear solve residue failed")
        for field in ("dual_cell_balance_scaled","dual_boundary_budget_scaled"):
            insist(float(row[field])
                   <=configuration["tolerances"]["dual_cell_balance_scaled_max"],
                   f"{case}: dual control-cell conservation failed")
        _,physical_rhs=analytic(n,angle)
        inside,method_details=reference_solve(n,angle,method,physical_rhs)
        metrics=face_metrics(inside,n,angle,method)
        for name in METRIC_FIELDS:
            approx(float(row[name]),metrics[name],case+"/"+name,float_tol)
        path=result/"raw"/case
        saved=json.loads((path/"run.json").read_text(encoding="utf-8"))
        insist(saved["article_id"]=="tutorial-007" and saved["experiment"]==STAGE
               and saved["case_id"]==case
               and saved["status"]=="solved"
               and saved["complete_physical_source_for_both_methods"] is True
               and saved["scientific_csv_sha256"]=={}
               and saved["config_sha256"]==sha256(config_path)
               and saved["source_sha256"]==run["source_sha256"],
               f"{case}: provenance, no-field contract or source differs")
        independent.append({"case_id":case,"reference":method_details,
                            "verified_T_error_l2":metrics["T_error_l2"],
                            "verified_qx_error_linf":metrics["q_x_error_linf"],
                            "verified_qy_error_linf":metrics["q_y_error_linf"]})
    real_csv=list(result.rglob("*.csv"))
    insist(len(real_csv)==2,"expected exactly fresh and comparison CSV")
    original28={
        *[r["case_id"] for r in previous["ET007-01"]],
        *[r["case_id"] for r in previous["ET007-02"]],
        *[r["case_id"] for r in previous["ET007-03"]],
        *[r["case_id"] for r in fresh],
    }
    insist(len(original28)==28,"entire article would not cover 28 unique solves")
    orders=[]
    for angle in ANGLES:
        for method in METHODS:
            group={int(r["N"]):r for r in view
                   if int(float(r["theta_deg"]))==angle and r["method"]==method}
            values={}
            for key in ("T_error_l2","T_error_linf",
                        "q_x_error_linf","q_y_error_linf"):
                x,y=float(group[64][key]),float(group[128][key])
                threshold=1000*np.finfo(float).eps
                values[key]=(float(math.log2(x/y))
                             if x>threshold and y>threshold
                             else "undefined_roundoff_floor")
            orders.append({"theta_deg":angle,"method":method,
                           "N0":64,"N1":128,"real_observed_orders":values})
    return {"status":"PASS","stage":STAGE,"article_id":"tutorial-007",
            "new_real_solve_cases":8,"reused_verified_case_rows":8,
            "angular_comparison_points":16,"article_unique_case_count":28,
            "full_all_four_stage_review_finished":False,
            "config_sha256":sha256(config_path),
            "production_source_commit":run.get("source_commit"),
            "production_solver_imported":False,
            "independent_methods":{
                "M0":"exact finite-grid DST-I",
                "M1":"matrix-free nine-point operator + CG"},
            "scientific_source_reuse_checked":True,
            "true_physical_tensor_source_used_for_both_methods":True,
            "internal_Omega_h_discrete_flux_budget_checked":True,
            "physical_Omega_outer_boundary_not_validated":True,
            "new_stage_summaries":independent,
            "comparisons_for_0_15_30_45":pairs,
            "angle_case_provenance":provenance,
            "refinement_orders_64_to_128":orders,
            "ET007_01_02_03_frozen_summary_sha256":SOURCE_HASHES,
            "science_accuracy_trends_not_forced":True}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--et01-summary",type=Path,required=True)
    parser.add_argument("--et02-summary",type=Path,required=True)
    parser.add_argument("--et03-summary",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("will not overwrite independent science")
    validated=verify(args.config.resolve(),args.results.resolve(),
                     args.et01_summary.resolve(),args.et02_summary.resolve(),
                     args.et03_summary.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(validated,ensure_ascii=False,indent=2,
                                   sort_keys=True,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("ET007_04_INDEPENDENT_PASS_NEW8_REUSED8",
          len(validated["new_stage_summaries"]),"new sparse systems",
          validated["angular_comparison_points"],"16-point comparison",
          flush=True)


if __name__=="__main__":
    main()
