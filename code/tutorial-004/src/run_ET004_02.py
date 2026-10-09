"""ET004-02: preregistered synthetic denoising benchmark; real output only."""
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

from diffusion_solver import advance_one_step
from image_inputs import make_synthetic_image, make_observation, input_sha256
from metrics import assess, edge_profile, PROFILE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "method", "t", "step", "rmse", "psnr_db", "rmse_flat",
    "rmse_edge", "edge_jump_ratio", "mean", "minimum", "maximum",
    "centered_ss",
)
PROFILE_FIELDS = ("i", "clean", "noisy", "heat", "regularized")
METHODS = ("heat", "regularized")
TIMES = (0, 2, 4, 8, 16, 32)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_config(cfg: dict) -> tuple[int, float, float, float, tuple[int, ...]]:
    if cfg.get("experiment_id") != "ET004-02":
        raise ValueError("wrong experiment")
    if cfg["status"] != "preregistered_before_formal_execution":
        raise ValueError("contract status changed")
    N = cfg["image"]["N"]
    h = float(cfg["image"]["h"])
    dt = float(cfg["solver"]["dt"])
    sigma = float(cfg["solver"]["sigma_pixels"])
    kappa = float(cfg["solver"]["kappa_intensity_per_pixel"])
    times = tuple(cfg["outputs"]["t"])
    if (N, h, dt, sigma, kappa, times) != (256, 1.0, 0.2, 1.5, 0.075, TIMES):
        raise ValueError("frozen geometry / diffusion parameters changed")
    if cfg["solver"]["methods"] != list(METHODS):
        raise ValueError("methods changed")
    if cfg["solver"]["gaussian_mode"] != "reflect" or cfg["solver"]["truncate"] != 4:
        raise ValueError("Gaussian regularization changed")
    if cfg["noise"] != {
        "kind": "iid_normal", "mean": 0, "std": 0.08,
        "bit_generator": "PCG64", "seed": 20261008, "no_clip": True
    }:
        raise ValueError("noise contract changed")
    if cfg["image"] != {
      "N": 256, "h": 1, "background": 0.20,
      "rectangle": {"i": [30, 110], "j": [40, 125], "brightness": 0.78},
      "circle": {"center_i": 181, "center_j": 174, "radius_inclusive": 35, "brightness": 0.58},
      "gradient": {"i": [150, 246], "j": [25, 90], "start": 0.25, "end": 0.70, "denominator": 95}
    }:
        raise ValueError("synthetic image geometry changed")
    if cfg["outputs"]["edge_profile_at"] != 16:
        raise ValueError("edge profile time changed")
    if cfg["outputs"]["flat_roi"] != {"i": [50, 90], "j": [65, 105]}:
        raise ValueError("flat ROI changed")
    if cfg["outputs"]["edge_roi"] != {"i": [24, 36], "j": [65, 105]}:
        raise ValueError("edge ROI changed")
    if cfg["outputs"]["jump"] != {"left_i": 29, "right_i": 30, "j": [65, 105], "denominator": 0.58}:
        raise ValueError("edge jump definition changed")
    if cfg["matched_flat_rmse"] != {"threshold_inclusive": 0.04, "candidates": list(TIMES)}:
        raise ValueError("matched-flat comparison was changed")
    if cfg["outputs"]["edge_x_indices"] != [20, 45]:
        raise ValueError("profile window changed")
    if cfg["encoding"] != {
       "dtype": "float64", "no_clipping_for_analysis": True,
       "snapshot": "npz_compressed_lossless", "csv": "utf-8"
    }:
        raise ValueError("output definition changed")
    if any(not math.isclose(round(t / dt) * dt, t, rel_tol=0, abs_tol=1e-12) for t in times):
        raise ValueError("fixed observation times must match steps exactly")
    if dt / (h*h) > 0.25:
        raise ValueError("CFL condition violated")
    return N, h, dt, sigma, times


def add_metrics(rows: list[dict], method: str, t: int, step: int,
                values: np.ndarray, reference: np.ndarray) -> dict:
    metrics = assess(values, reference)
    row = {"method": method, "t": t, "step": step, **metrics}
    rows.append(row)
    return row


def validate_step(
    values: np.ndarray, *, previous_variance: float, initial_mean: float,
    initial_min: float, initial_max: float, thresholds: dict
) -> tuple[float, float, float, float]:
    if not np.isfinite(values).all():
        raise FloatingPointError("ET004-02 nonfinite state")
    mean = float(np.mean(values))
    min_value, max_value = float(np.min(values)), float(np.max(values))
    variance = float(np.sum((values-mean)**2))
    mean_drift = abs(mean-initial_mean)
    range_violation = max(0.0, initial_min-min_value, max_value-initial_max)
    variance_increase = max(0.0, variance-previous_variance)
    if mean_drift > thresholds["max_mean_drift"]:
        raise AssertionError("ET004-02 mean conservation failed")
    if range_violation > thresholds["max_range_violation"]:
        raise AssertionError("ET004-02 maximum principle failed")
    if variance_increase > thresholds["max_centered_ss_increase"]:
        raise AssertionError("ET004-02 centered variance increased")
    return variance,mean_drift,range_violation,variance_increase


def run(cfg_path: Path, out: Path, source_sha: str) -> None:
    if len(source_sha) != 40:
        raise ValueError("source SHA must be an exact 40-character commit")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    N,h,dt,sigma,times = checked_config(cfg)
    kappa = float(cfg["solver"]["kappa_intensity_per_pixel"])
    limit = cfg["invariants"]
    clean = make_synthetic_image(N)
    noisy = make_observation(clean)
    if clean.shape != (256,256) or clean.dtype != np.float64 or noisy.dtype != np.float64:
        raise AssertionError("wrong frozen image dtype or dimensions")
    if not (np.isfinite(clean).all() and np.isfinite(noisy).all()):
        raise AssertionError("nonfinite initial array")
    if not (clean[50,20] == 0.20 and clean[70,50] == 0.78 and
            clean[174,181] == 0.58 and clean[40,150] == 0.25):
        raise AssertionError("reference geometry anchor values changed")
    if np.array_equal(clean, noisy):
        raise AssertionError("no noise generated")
    if np.min(noisy) >= 0 and np.max(noisy) <= 1:
        # This is only a diagnostic statement; it is NOT an acceptance condition.
        print("ET004-02 NOTE: input may happen to remain in [0,1]", flush=True)

    output_steps={int(round(t/dt)):t for t in times}
    last_step=int(round(times[-1]/dt))
    initial_mean=float(np.mean(noisy))
    initial_min=float(np.min(noisy))
    initial_max=float(np.max(noisy))
    initial_variance=float(np.sum((noisy-initial_mean)**2))
    snapshot_arrays={"clean":clean.copy(), "noisy":noisy.copy()}
    rows=[]
    add_metrics(rows,"noisy",0,0,noisy,clean)
    peaks={}
    last_diagnostics={}
    print("ET004-02 INPUT clean_sha256="+input_sha256(clean)+
          " noisy_sha256="+input_sha256(noisy), flush=True)
    for method in METHODS:
        current=noisy.copy()
        previous_var=initial_variance
        max_drift=max_range=max_variance_increase=0.0
        for step in range(1,last_step+1):
            current=advance_one_step(
                current,dt=dt,h=h,method=method,sigma=sigma,kappa=kappa
            )
            previous_var, drift, rv, inc=validate_step(
                current, previous_variance=previous_var,
                initial_mean=initial_mean, initial_min=initial_min,
                initial_max=initial_max,thresholds=limit
            )
            max_drift=max(max_drift,drift)
            max_range=max(max_range,rv)
            max_variance_increase=max(max_variance_increase,inc)
            if step in output_steps:
                t=output_steps[step]
                snapshot_arrays[f"{method}_t{t:03d}"]=current.copy()
                row=add_metrics(rows,method,t,step,current,clean)
                print(f"ET004-02 {method} t={t} RMSE={row['rmse']:.15e}"
                      f" PSNR={row['psnr_db']:.10f} flat={row['rmse_flat']:.15e}"
                      f" edge={row['rmse_edge']:.15e} J={row['edge_jump_ratio']:.12f}",flush=True)
        peaks[method]={
          "max_mean_drift":max_drift,
          "max_range_violation":max_range,
          "max_centered_ss_increase":max_variance_increase,
          "final_mean":float(np.mean(current)),
          "final_centered_ss":float(np.sum((current-np.mean(current))**2)),
          "steps":last_step
        }
        last_diagnostics[method]=current.copy()
    if len(rows) != 1+len(METHODS)*(len(times)-1):
        raise AssertionError("incorrect number of frozen snapshot rows")

    profile_t=cfg["outputs"]["edge_profile_at"]
    profiles={
       "clean":edge_profile(clean), "noisy":edge_profile(noisy),
       "heat":edge_profile(snapshot_arrays[f"heat_t{profile_t:03d}"]),
       "regularized":edge_profile(snapshot_arrays[f"regularized_t{profile_t:03d}"])
    }
    if len(PROFILE_COLUMNS)!=26:
        raise AssertionError("changed edge profile coordinates")
    matched={}
    threshold=float(cfg["matched_flat_rmse"]["threshold_inclusive"])
    for method in METHODS:
        candidates=[row for row in rows if row["method"]==method and row["rmse_flat"]<=threshold]
        # Noisy baseline t=0 is shared, not listed for either method.
        baseline=rows[0]
        if baseline["rmse_flat"]<=threshold:
            first=baseline
        else:
            first=candidates[0] if candidates else None
        matched[method] = None if first is None else {
             "t":first["t"],"rmse_flat":first["rmse_flat"],
             "rmse_edge":first["rmse_edge"],"edge_jump_ratio":first["edge_jump_ratio"]
        }
    matched_comparison = None
    if matched["heat"] is not None and matched["regularized"] is not None:
        matched_comparison={
            "heat_t":matched["heat"]["t"],"regularized_t":matched["regularized"]["t"],
            "edge_rmse_regularized_minus_heat":
                matched["regularized"]["rmse_edge"]-matched["heat"]["rmse_edge"],
            "edge_jump_regularized_minus_heat":
                matched["regularized"]["edge_jump_ratio"]-matched["heat"]["edge_jump_ratio"]
        }
    print("ET004-02 MATCHED_FLAT "+json.dumps({
       "threshold":threshold,"matches":matched,"comparison":matched_comparison
    },sort_keys=True),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out/"edge_profiles.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("i","clean","noisy","heat","regularized"),lineterminator="\n")
        writer.writeheader()
        for k,i in enumerate(PROFILE_COLUMNS):
            writer.writerow({"i":int(i),**{name:float(profile[k]) for name,profile in profiles.items()}})
    np.savez_compressed(out/"snapshots.npz",**snapshot_arrays)
    checks={
       "experiment_id":"ET004-02",
       "status":"PASS",
       "scientific_scope":"synthetic image with known reference; two PDE methods; no real camera",
       "source_commit":source_sha,
       "runner_type":"GitHub self-hosted",
       "contract":cfg,
       "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,
                  "scipy":metadata.version("scipy"),"platform":platform.platform()},
       "input":{"clean_sha256_float64_c":input_sha256(clean),
                "noisy_sha256_float64_c":input_sha256(noisy),
                "clean_range":[float(clean.min()),float(clean.max())],
                "noisy_range":[initial_min,initial_max],
                "initial_mean":initial_mean,"initial_centered_ss":initial_variance},
       "source_hashes":{
          name:digest(ROOT/"src"/name) for name in
          ("diffusion_solver.py","image_inputs.py","metrics.py","run_ET004_02.py")
       },
       "config_sha256":digest(cfg_path),
       "outputs_sha256":{
          name:digest(out/name) for name in ("metrics.csv","edge_profiles.csv","snapshots.npz")
       },
       "number_of_metric_records":len(rows),
       "number_of_profile_records":len(PROFILE_COLUMNS),
       "number_of_snapshot_arrays":len(snapshot_arrays),
       "diagnostics":peaks,
       "matched_flat":{"threshold":threshold,"methods":matched,"comparison":matched_comparison},
       "passes":{
          "input_unclipped":True,
          "shared_initial_image_sha":input_sha256(noisy),
          "invariants_every_step":True,
          "scheduled_t_only":True,
          "linear_heat_and_regularized_computed":True,
          "nonlinear_method_not_assumed_better":True
       },
       "not_covered":["ET004-03 parameter scan","ET004-04 timestep convergence",
                      "ET004-05 real photograph","FT004 figure rendering"]
    }
    (out/"checks.json").write_text(json.dumps(checks,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    print("ET004-02 FORMAL_GATE PASS INPUT_SHA="+input_sha256(noisy)+
          " METRICS_SHA="+checks["outputs_sha256"]["metrics.csv"]+
          " SNAPSHOTS_SHA="+checks["outputs_sha256"]["snapshots.npz"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,default=ROOT/"config/ET004-02.json")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--source-sha",type=str,default=os.getenv("GITHUB_SHA","UNKNOWN"))
    args=ap.parse_args()
    run(args.config,args.out,args.source_sha)
