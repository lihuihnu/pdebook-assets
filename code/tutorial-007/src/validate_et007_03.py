#!/usr/bin/env python3
"""Independent ET007-03 checker: NEVER imports the production PDE solver.

For M0, independently diagonalize the exact finite-difference Dirichlet
operator by an orthonormal two-dimensional DST-I. For M1, independently
reconstruct the physical nine-point stencil as a matrix-free LinearOperator
and solve it by CG. Both differ from production's assembled CSR+spsolve.
Check all 8 summary rows and the actual stored 64/128 node/face CSVs.
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


GRIDS = (16, 32, 64, 128)
METHODS = ("M0", "M1")
KXX = 5.5
KXY = 4.5
KYY = 5.5
FULL_CASES = tuple(f"sinesine-p45-N{n}-{method}"
                   for n in GRIDS for method in METHODS)
METRICS = ("T_error_l2", "T_error_linf", "q_x_error_linf",
           "q_y_error_linf", "dual_cell_balance_linf",
           "dual_cell_balance_scaled", "dual_boundary_budget_defect",
           "dual_boundary_budget_scaled")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def close(actual: float, expected: float, name: str, tolerance: float) -> None:
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= tolerance*max(1.,abs(expected)),
            f"{name}: got {actual:.17g}, expected {expected:.17g}, tolerance {tolerance:.3g}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def analytic_values(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Independently differentiate the exact FULL tensor sine solution."""
    h=1./n
    line=np.arange(1,n)*h
    x,y=np.meshgrid(line,line,indexing="ij")
    exact=np.sin(math.pi*x)*np.sin(math.pi*y)
    source=math.pi**2*(11.0*exact
                       -9.0*np.cos(math.pi*x)*np.cos(math.pi*y))
    return exact,source


def independent_operator(v: np.ndarray, n: int, method: str) -> np.ndarray:
    """Interior Dirichlet stencil without sparse matrix assembly."""
    h=1./n
    u=np.zeros((n+1,n+1),dtype=np.float64)
    u[1:n,1:n]=np.asarray(v).reshape((n-1,n-1))
    mid=u[1:n,1:n]
    out=-(KXX*(u[2:,1:n]-2*mid+u[:-2,1:n])
          +KYY*(u[1:n,2:]-2*mid+u[1:n,:-2]))/(h*h)
    if method=="M1":
        out-=KXY*(u[2:,2:]-u[2:,:-2]
                    -u[:-2,2:]+u[:-2,:-2])/(2*h*h)
    elif method!="M0":
        raise ValueError("unexpected ET007-03 method")
    return out.ravel()


def spectral_M0(n: int, physical_source: np.ndarray) -> tuple[np.ndarray, dict]:
    """Exactly diagonalize the M0 five-point discrete operator via DST-I."""
    m=n-1
    eig_idx=np.arange(1,n)
    hx=np.sin(math.pi*eig_idx/(2*n))**2
    eigenvalues=4*n*n*(KXX*hx[:,None]+KYY*hx[None,:])
    hat=dstn(physical_source,type=1,norm="ortho")
    u=idstn(hat/eigenvalues,type=1,norm="ortho")
    residual=independent_operator(u.ravel(),n,"M0").reshape((m,m))-physical_source
    return u, {"method":"DST-I/finite-grid eigenvalues",
               "scaled_residual_linf":float(np.max(np.abs(residual))/
                               max(1.,float(np.max(np.abs(physical_source)))))}


def matrix_free_M1(n: int, physical_source: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve the cross-diffusion stencil with independent matrix-free CG."""
    m=n-1
    size=m*m
    op=LinearOperator((size,size),dtype=np.float64,
                      matvec=lambda v:independent_operator(v,n,"M1"))
    # Constant diagonal Jacobi; no reuse of the production sparse matrix.
    inv_diag=1./(2*(KXX+KYY)*n*n)
    precondition=LinearOperator((size,size),dtype=np.float64,
                                matvec=lambda v:inv_diag*np.asarray(v))
    count=[0]
    def tick(_x):
        count[0]+=1
    x,info=cg(op,physical_source.ravel(),M=precondition,
              rtol=1e-13,atol=0,maxiter=4000,callback=tick)
    require(info==0 and np.all(np.isfinite(x)),
            f"N={n}: matrix-free M1 CG failed, info={info}, iters={count[0]}")
    residual=independent_operator(x,n,"M1")-physical_source.ravel()
    scaled=float(np.max(np.abs(residual))/
                 max(1.,float(np.max(np.abs(physical_source)))))
    require(scaled<=1e-10,
            f"N={n}: independent nine-point operator residual too large {scaled}")
    return x.reshape((m,m)), {
        "method":"matrix-free independent nine-point CG",
        "independent_iterations":count[0],
        "scaled_residual_linf":scaled,
    }


def independent_face_fields(u: np.ndarray, method: str) -> tuple[np.ndarray,np.ndarray]:
    """Independent from production: share each dual face once."""
    n=u.shape[0]-1
    h=1./n
    qx=-KXX*(u[1:,1:n]-u[:-1,1:n])/h
    qy=-KYY*(u[1:n,1:]-u[1:n,:-1])/h
    if method=="M1":
        # average the two neighbouring centred tangential derivatives
        tang_y=(u[1:,2:]-u[1:,:-2]+u[:-1,2:]-u[:-1,:-2])/(4*h)
        tang_x=(u[2:,1:]-u[:-2,1:]+u[2:,:-1]-u[:-2,:-1])/(4*h)
        qx-=KXY*tang_y
        qy-=KXY*tang_x
    return qx,qy


def independent_diagnostics(inside: np.ndarray, n: int,
                            method: str) -> tuple[dict[str,float],np.ndarray,np.ndarray,np.ndarray]:
    h=1./n
    u=np.zeros((n+1,n+1),dtype=np.float64)
    u[1:n,1:n]=inside
    exact,source=analytic_values(n)
    delta=inside-exact
    qx,qy=independent_face_fields(u,method)
    xv=(np.arange(n)+.5)*h
    yv=np.arange(1,n)*h
    true_qx=-math.pi*(KXX*np.cos(math.pi*xv)[:,None]*
                np.sin(math.pi*yv)[None,:] +
                KXY*np.sin(math.pi*xv)[:,None]*
                np.cos(math.pi*yv)[None,:])
    xh=np.arange(1,n)*h
    yh=(np.arange(n)+.5)*h
    true_qy=-math.pi*(KXY*np.cos(math.pi*xh)[:,None]*
                np.sin(math.pi*yh)[None,:] +
                KYY*np.sin(math.pi*xh)[:,None]*
                np.cos(math.pi*yh)[None,:])
    balance=h*(qx[1:]-qx[:-1]+qy[:,1:]-qy[:,:-1])-h*h*source
    flux_scale=max(1.,float(np.max(np.abs(h*qx))),
                      float(np.max(np.abs(h*qy))),h*h*float(np.max(np.abs(source))))
    budget=float(h*(np.sum(qx[-1])-np.sum(qx[0])
                 +np.sum(qy[:,-1])-np.sum(qy[:,0]))-h*h*np.sum(source))
    data={
        "T_error_l2":float(np.sqrt(h*h*np.sum(delta*delta))),
        "T_error_linf":float(np.max(np.abs(delta))),
        "q_x_error_linf":float(np.max(np.abs(qx-true_qx))),
        "q_y_error_linf":float(np.max(np.abs(qy-true_qy))),
        "dual_cell_balance_linf":float(np.max(np.abs(balance))),
        "dual_cell_balance_scaled":float(np.max(np.abs(balance))/flux_scale),
        "dual_boundary_budget_defect":budget,
        "dual_boundary_budget_scaled":abs(budget)/flux_scale,
    }
    return data,u,qx,qy


def saved_fields(output: Path,case: str,n: int,
                 independent_u: np.ndarray,
                 independent_qx: np.ndarray,
                 independent_qy: np.ndarray,
                 tolerance: float) -> dict:
    """Inspect all CSV rows, physical positions, and independent scheme results."""
    raw=output/"raw"/case
    temperature=read_csv(raw/"temperature.csv")
    vertical=read_csv(raw/"flux_x.csv")
    horizontal=read_csv(raw/"flux_y.csv")
    require(len(temperature)==(n+1)**2
            and len(vertical)==n*(n-1)
            and len(horizontal)==n*(n-1),
            f"{case}: incomplete saved nodal/face samples")
    h=1./n
    restored_u=np.empty((n+1,n+1))
    restored_x=np.empty((n,n-1))
    restored_y=np.empty((n-1,n))
    for v,(i,j) in zip(temperature,
                       ((i,j) for i in range(n+1) for j in range(n+1)),strict=True):
        require((int(v["i"]),int(v["j"]))==(i,j),f"{case}: bad node order")
        close(float(v["x"]),i*h,"node x",1e-13)
        close(float(v["y"]),j*h,"node y",1e-13)
        restored_u[i,j]=float(v["u"])
    for v,(i,j) in zip(vertical,
                       ((i,j) for i in range(n) for j in range(1,n)),strict=True):
        require((int(v["i"]),int(v["j"]))==(i,j),f"{case}: bad qx order")
        close(float(v["x"]),(i+.5)*h,"vertical face x",1e-13)
        close(float(v["y"]),j*h,"vertical face y",1e-13)
        restored_x[i,j-1]=float(v["q_x"])
    for v,(i,j) in zip(horizontal,
                       ((i,j) for i in range(1,n) for j in range(n)),strict=True):
        require((int(v["i"]),int(v["j"]))==(i,j),f"{case}: bad qy order")
        close(float(v["x"]),i*h,"horizontal face x",1e-13)
        close(float(v["y"]),(j+.5)*h,"horizontal face y",1e-13)
        restored_y[i-1,j]=float(v["q_y"])
    require(np.all(np.isfinite(restored_u)) and np.all(np.isfinite(restored_x))
            and np.all(np.isfinite(restored_y)),f"{case}: nonfinite raw field")
    node_diff=float(np.max(np.abs(restored_u-independent_u)))
    flux_x_diff=float(np.max(np.abs(restored_x-independent_qx)))
    flux_y_diff=float(np.max(np.abs(restored_y-independent_qy)))
    # Face gradients magnify independently bounded nodal solver roundoff by O(1/h).
    require(node_diff <= tolerance*max(1.,float(np.max(np.abs(independent_u)))),
            f"{case}: production nodes disagree with independent spectral/CG solve: {node_diff}")
    for what,value,other in [("qx",flux_x_diff,independent_qx),
                             ("qy",flux_y_diff,independent_qy)]:
        require(value<=max(2e-8,tolerance*max(1.,float(np.max(np.abs(other))))*10),
                f"{case}: saved {what} differs from the independent solve: {value}")
    # Stronger internal consistency: derive each stored flux from the STORED
    # nodes by a separately coded face stencil; compare to original face files.
    from_nodes_x,from_nodes_y=independent_face_fields(
        restored_u,case.rsplit("-",1)[1])
    require(float(np.max(np.abs(restored_x-from_nodes_x)))<=2e-10,
            f"{case}: archived qx CSV is inconsistent with archived T")
    require(float(np.max(np.abs(restored_y-from_nodes_y)))<=2e-10,
            f"{case}: archived qy CSV is inconsistent with archived T")
    return {"case_id":case,"node_count":len(temperature),
            "dual_face_count":len(vertical)+len(horizontal),
            "max_difference_from_independent_solution":node_diff,
            "max_qx_difference_from_independent_solution":flux_x_diff,
            "max_qy_difference_from_independent_solution":flux_y_diff}


def metric_orders(table: list[dict]) -> list[dict]:
    """Do not turn near-roundoff ratios into fictitious convergence orders."""
    threshold=1000*np.finfo(float).eps
    orders=[]
    for method in METHODS:
        rr={int(r["N"]):r for r in table if r["method"]==method}
        for n in GRIDS[:-1]:
            metrics={}
            for name in ("T_error_l2","T_error_linf",
                         "q_x_error_linf","q_y_error_linf"):
                e0=float(rr[n][name])
                e1=float(rr[2*n][name])
                metrics[name]=(float(math.log2(e0/e1))
                               if e0>threshold and e1>threshold
                               else "undefined_roundoff_floor")
            orders.append({"method":method,"N_coarse":n,"N_fine":2*n,
                           "observed_orders":metrics})
    return orders


def verify(config_path: Path,output: Path) -> dict:
    config=json.loads(config_path.read_text(encoding="utf-8"))
    require(config["status"]=="science_parameters_frozen"
            and config["design_version"]=="v0.2"
            and config["tensor_principal"]==[10.,1.]
            and config["planned_unique_solve_count"]==28,
            "not the pre-registered physical problem")
    stage=next(s for s in config["experiment_sets"] if s["id"]=="ET007-03")
    require(stage["theta_deg"]==45. and stage["field"]=="sinesine"
            and stage["grids_N"]==list(GRIDS)
            and stage["methods"]==list(METHODS)
            and stage["save_full_fields_N"]==[64,128],
            "ET007-03 case matrix modified after freeze")
    run=json.loads((output/"run.json").read_text(encoding="utf-8"))
    require(run["article_id"]=="tutorial-007" and run["experiment"]=="ET007-03"
            and run["status"]=="completed" and run["full_tensor_source_shared"]
            and run["executed_case_count"]==8
            and run["planned_full_case_count"]==28
            and not run["complete_28_case_program"],
            "stage identity or true-source contract violated")
    require(run["run_case_ids"]==list(FULL_CASES),"wrong stage case identities")
    require(run["config_sha256"]==sha256(config_path),"configuration changed")
    source_root=Path(__file__).resolve().parents[1]
    for path,expected in run["source_sha256"].items():
        require(sha256(source_root/path)==expected,
                "production source identity drift: "+path)
    for path,expected in run["scientific_sha256"].items():
        require(sha256(output/path)==expected,
                "frozen result byte identity drift: "+path)
    summary=read_csv(output/"summary.csv")
    require(len(summary)==8 and [row["case_id"] for row in summary]==list(FULL_CASES),
            "missing, reordered, or duplicate ET007-03 result row")
    expected_csv=1+2*3*2  # 1 summary plus N64/N128 x 2 methods x 3 raw CSV
    require(len(list(output.rglob("*.csv")))==expected_csv,
            "unexpected or missing planned CSV files")
    independent=[]
    observed_fields=[]
    tolerance=float(config["tolerances"]["independent_check_linf_relative_max"])
    require(tolerance==1e-9,"independent tolerance not frozen")
    for record in summary:
        n=int(record["N"]); method=record["method"]
        case=record["case_id"]
        require(record["field"]=="sinesine" and record["experiment"]=="ET007-03"
                and record["solver_status"]=="solved"
                and float(record["theta_deg"])==45.0,
                f"{case}: unexpected physical case")
        require(int(record["n_unknowns"])==(n-1)**2,"wrong matrix dimension")
        m=n-1
        expected_nnz=(5*m*m-4*m) if method=="M0" else (9*m*m-12*m+4)
        require(int(record["nnz"])==expected_nnz,
                f"{case}: incorrect matrix stencil structure")
        for k,v in (("K_xx",KXX),("K_xy",KXY),("K_yy",KYY)):
            close(float(record[k]),v,k+" "+case,1e-13)
        require(float(record["linear_residual_linf_relative"])
                <=config["tolerances"]["linear_residual_linf_relative_max"],
                f"{case}: linear solver residual exceeds frozen threshold")
        for key in ("dual_cell_balance_scaled","dual_boundary_budget_scaled"):
            require(float(record[key])
                    <=config["tolerances"]["dual_cell_balance_scaled_max"],
                    f"{case}: Omega_h conservation exceeded frozen limit")
        exact,source=analytic_values(n)
        if method=="M0":
            inside,solver=spectral_M0(n,source)
        else:
            inside,solver=matrix_free_M1(n,source)
        diagnostics,reference_u,reference_qx,reference_qy=independent_diagnostics(
            inside,n,method)
        for key in METRICS:
            close(float(record[key]),diagnostics[key],case+"/"+key,tolerance)
        saved=json.loads((output/"raw"/case/"run.json").read_text(encoding="utf-8"))
        require(saved["case_id"]==case and saved["status"]=="solved"
                and saved["same_full_continuous_source_for_both_methods"] is True
                and saved["config_sha256"]==sha256(config_path),
                f"{case}: case/source identity invalid")
        require(saved["source_sha256"]==run["source_sha256"],
                f"{case}: differing production source hash")
        for name,expected in saved["scientific_csv_sha256"].items():
            require(sha256(output/"raw"/case/name)==expected,
                    f"{case}: original field byte hash differs")
        if n in (64,128):
            require(set(saved["scientific_csv_sha256"])==
                    {"temperature.csv","flux_x.csv","flux_y.csv"},
                    f"{case}: one or more planned physical fields missing")
            observed_fields.append(saved_fields(output,case,n,reference_u,
                                               reference_qx,reference_qy,tolerance))
        else:
            require(not saved["scientific_csv_sha256"],
                    f"{case}: unplanned raw fields included")
        independent.append({"case_id":case,"reference_solver":solver,
                            "reference_T_error_l2":diagnostics["T_error_l2"],
                            "reference_qx_error_linf":diagnostics["q_x_error_linf"],
                            "reference_qy_error_linf":diagnostics["q_y_error_linf"]})
    return {"status":"PASS","article_id":"tutorial-007","experiment":"ET007-03",
            "actual_cases":8,"planned_full_cases":28,
            "scientific_csv_count":expected_csv,
            "frozen_config_sha256":sha256(config_path),
            "production_source_commit":run.get("source_commit"),
            "production_solver_imported":False,
            "M0_reference":"orthonormal 2D DST-I eigenvalue solution",
            "M1_reference":"matrix-free CG on independently coded nine-point Dirichlet operator",
            "same_true_full_tensor_source":True,
            "metrics_compared_independently_for_all_eight_cases":True,
            "raw_field_reviews":observed_fields,
            "independent_solutions":independent,
            "observed_refinement_orders_from_real_summary":metric_orders(summary),
            "scope":"internal Omega_h dual conservation, not full physical boundary flux",
            "convergence_not_forced":True}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--results",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to replace an independent scientific review")
    verdict=verify(args.config.resolve(),args.results.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(verdict,indent=2,ensure_ascii=False,
                                    sort_keys=True,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("ET007_03_INDEPENDENT_SCIENCE_PASS",verdict["actual_cases"],
          "real systems",len(verdict["raw_field_reviews"]),
          "complete saved fields",flush=True)


if __name__=="__main__":
    main()
