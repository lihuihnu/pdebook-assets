"""ET004-01: preregistered Neumann heat-equation control, not photo data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from importlib import metadata

import numpy as np
from diffusion_solver import _face_coefficients, advance_one_step

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "dt", "steps", "r", "lambda_h", "cells", "interior_x_faces",
    "interior_y_faces", "boundary_faces", "constant_max_deviation",
    "mean_max_deviation", "range_violation_max", "spatial_eigenpair_residual",
    "modal_euler_formula_max_deviation", "rmse_vs_semidiscrete_exact",
    "l2h_vs_semidiscrete_exact", "linf_vs_semidiscrete_exact",
    "observed_order_rmse", "observed_order_linf"
)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def maximum_abs(x: np.ndarray) -> float:
    if not np.isfinite(x).all():
        raise FloatingPointError("nonfinite values in ET004-01")
    return float(np.max(np.abs(x)))

def face_divergence(U: np.ndarray, h: float) -> np.ndarray:
    """Independent assembly of the four-face Neumann Laplacian."""
    lap = np.zeros_like(U)
    dx = (U[:, 1:] - U[:, :-1]) / (h*h)
    dy = (U[1:, :] - U[:-1, :]) / (h*h)
    lap[:, :-1] += dx
    lap[:, 1:] -= dx
    lap[:-1, :] += dy
    lap[1:, :] -= dy
    return lap

def run(config_path: Path, output: Path, source_sha: str) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["experiment_id"] != "ET004-01" or cfg["status"] != "preregistered_before_formal_run":
        raise ValueError("unknown ET004-01 configuration")
    N,h,T = cfg["N"],float(cfg["h"]),float(cfg["T"])
    dts = [float(x) for x in cfg["dt_levels"]]
    if (N,h,T,dts)!=(64,1.0,8.0,[0.2,0.1,0.05,0.025]):
        raise ValueError("frozen grid or time levels modified")
    if (cfg["constant"],cfg["cosine_base"],cfg["cosine_amplitude"])!=(0.42,0.5,0.1):
        raise ValueError("frozen Neumann initial conditions modified")
    limits=cfg["thresholds"]
    x=np.arange(N,dtype=np.float64)+0.5
    profile=np.cos(np.pi*x/N)
    mode=np.tile(profile,(N,1))
    initial=0.5+0.1*mode
    constant=np.full((N,N),0.42,dtype=np.float64)
    eig=-4.0*math.sin(math.pi/(2*N))**2/(h*h)
    ref=0.5+0.1*math.exp(eig*T)*mode
    eigen_residual=maximum_abs(face_divergence(initial,h)-eig*(initial-0.5))
    if eigen_residual>limits["spatial_eigenpair_residual_max"]:
        raise AssertionError("Neumann spatial eigenpair control failed")
    ax,ay=_face_coefficients(initial,method="heat",sigma=1.5,kappa=0.075,h=h)
    if ax.shape!=(N,N-1) or ay.shape!=(N-1,N) or not (np.all(ax==1) and np.all(ay==1)):
        raise AssertionError("heat shared-face coefficients are not unity")
    rows=[]
    previous=None
    global_constant=global_mass=global_euler_formula=0.0
    for dt in dts:
        steps=round(T/dt)
        if steps<=0 or not math.isclose(steps*dt,T,rel_tol=0,abs_tol=1e-12):
            raise ValueError("T is not an exact number of steps")
        Uc=constant.copy()
        Um=initial.copy()
        max_constant=max_mass=max_range=0.0
        for _ in range(steps):
            Uc=advance_one_step(Uc,dt=dt,h=h,method="heat")
            Um=advance_one_step(Um,dt=dt,h=h,method="heat")
            max_constant=max(max_constant,maximum_abs(Uc-constant))
            max_mass=max(max_mass,abs(float(Uc.mean()-constant.mean())),abs(float(Um.mean()-initial.mean())))
            max_range=max(max_range,max(0.0,float(initial.min()-Um.min())),max(0.0,float(Um.max()-initial.max())))
        modal_euler=0.5+0.1*(1+eig*dt)**steps*mode
        euler_formula=maximum_abs(Um-modal_euler)
        difference=Um-ref
        rmse=float(np.sqrt(np.mean(difference*difference)))
        l2h=float(h*np.sqrt(np.sum(difference*difference)))
        linf=maximum_abs(difference)
        p_rmse=p_linf=""
        if previous is not None:
            if min(previous["rmse"],previous["linf"],rmse,linf)<=0:
                raise AssertionError("cannot compute observed order from zero error")
            denominator=math.log(previous["dt"]/dt)
            p_rmse=math.log(previous["rmse"]/rmse)/denominator
            p_linf=math.log(previous["linf"]/linf)/denominator
        row={
          "dt":dt,"steps":steps,"r":dt/(h*h),"lambda_h":eig,
          "cells":N*N,"interior_x_faces":N*(N-1),
          "interior_y_faces":N*(N-1),"boundary_faces":4*N,
          "constant_max_deviation":max_constant,
          "mean_max_deviation":max_mass,"range_violation_max":max_range,
          "spatial_eigenpair_residual":eigen_residual,
          "modal_euler_formula_max_deviation":euler_formula,
          "rmse_vs_semidiscrete_exact":rmse,
          "l2h_vs_semidiscrete_exact":l2h,
          "linf_vs_semidiscrete_exact":linf,
          "observed_order_rmse":p_rmse,"observed_order_linf":p_linf}
        rows.append(row)
        previous={"dt":dt,"rmse":rmse,"linf":linf}
        global_constant=max(global_constant,max_constant)
        global_mass=max(global_mass,max_mass)
        global_euler_formula=max(global_euler_formula,euler_formula)
        print(f"ET004-01 dt={dt} steps={steps} RMSE={rmse:.15e} "
              f"L2h={l2h:.15e} Linf={linf:.15e} p_RMSE={p_rmse} p_Linf={p_linf}",flush=True)

    reject=False
    try:
        advance_one_step(constant,dt=0.26,h=h,method="heat")
    except ValueError:
        reject=True
    if not reject:
        raise AssertionError("r>1/4 was not rejected")
    allowed=advance_one_step(constant,dt=0.25,h=h,method="heat")
    if maximum_abs(allowed-constant)>limits["constant_max_deviation"]:
        raise AssertionError("r=1/4 endpoint failed")
    if global_constant>limits["constant_max_deviation"] or global_mass>limits["mean_max_deviation"]:
        raise AssertionError("constant or mean conservation regression")
    if global_euler_formula>limits["modal_euler_formula_max_deviation"]:
        raise AssertionError("numerical mode disagrees with Euler amplification")
    for row in rows[-2:]:
        for name in ("observed_order_rmse","observed_order_linf"):
            if not limits["last_two_observed_orders_min"]<=row[name]<=limits["last_two_observed_orders_max"]:
                raise AssertionError(f"unacceptable convergence order: {name}, dt={row['dt']}")

    output.mkdir(parents=True,exist_ok=True)
    csv_file=output/"control_errors.csv"
    with csv_file.open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    checks={
      "experiment_id":"ET004-01","status":"PASS",
      "runner_type":"GitHub self-hosted","source_commit":source_sha,
      "scope":"linear heat, Neumann, constant and analytic semidiscrete cosine only",
      "reference":{"type":"analytic semidiscrete exact time evolution",
                   "lambda_h":eig,"spatial_eigenpair_residual_max":eigen_residual,
                   "formula":"0.5+0.1*exp(lambda_h*T)*cos(pi*(i+0.5)/N)"},
      "config":cfg,
      "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,
                 "scipy":metadata.version("scipy"),"platform":platform.platform()},
      "geometry":{"cells":N*N,"interior_x_faces":N*(N-1),
                  "interior_y_faces":N*(N-1),"boundary_faces":4*N},
      "checks":{"constant_max_deviation":global_constant,
                "mean_max_deviation":global_mass,
                "mode_euler_formula_max_deviation":global_euler_formula,
                "rejected_r_above_quarter":reject,
                "accepted_r_equal_quarter":True,
                "interior_face_coefficients_exactly_one":True,
                "last_two_orders_rmse":[rows[-2]["observed_order_rmse"],rows[-1]["observed_order_rmse"]],
                "last_two_orders_linf":[rows[-2]["observed_order_linf"],rows[-1]["observed_order_linf"]]},
      "provenance_sha256":{
         "config":digest(config_path),"solver":digest(ROOT/"src/diffusion_solver.py"),
         "driver":digest(Path(__file__).resolve()),"control_errors_csv":digest(csv_file),
         "initial_constant_float64":hashlib.sha256(constant.tobytes()).hexdigest(),
         "initial_cosine_float64":hashlib.sha256(initial.tobytes()).hexdigest()},
      "records_count":len(rows),"not_covered":["nonlinear denoising","true photograph","ET004-02..05"]
    }
    checks_path=output/"checks.json"
    checks_path.write_text(json.dumps(checks,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(f"ET004-01 FORMAL_GATE PASS; CSV_SHA256={digest(csv_file)} CHECKS_SHA256={digest(checks_path)}",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,default=ROOT/"config/ET004-01.json")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--source-sha",default=os.getenv("GITHUB_SHA","UNKNOWN"))
    args=ap.parse_args()
    run(args.config,args.out,args.source_sha)
