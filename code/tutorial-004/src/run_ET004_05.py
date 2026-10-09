"""ET004-05 real CC0 photograph + fixed synthetic Gaussian noise benchmark.

A verified scikit-image==0.25.2 camera.png fixture is a prerequisite. PDE
solvers depend only on NumPy/SciPy; the preparer owns scikit-image/Pillow.
Every emitted metric is computed from unclipped float64 arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image

from diffusion_solver import advance_one_step, _face_coefficients

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("heat", "regularized")
TIMES = (0, 2, 4, 8, 16, 32)
METRIC_COLUMNS = ("method","t","step","rmse","psnr_db","mean","minimum","maximum","centered_ss")
STEP_COLUMNS = (
    "method","step","t","mean","mean_drift","minimum","maximum",
    "range_violation","centered_ss","centered_ss_increase",
    "flux_update_residual","minimum_face_diffusivity","maximum_face_diffusivity"
)


def sha_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_float64(arr:np.ndarray)->str:
    if arr.dtype != np.float64 or arr.shape != (256,256):
        raise TypeError("array must be float64 256x256")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def require_parameters(cfg:dict)->tuple[float,float,float,int]:
    if (
        cfg.get("experiment_id") != "ET004-05"
        or cfg.get("status") != "preregistered_before_formal_execution"
        or cfg.get("scientific_contract") != "notes/tutorial-004/experiment-spec.md"
    ):
        raise ValueError("ET004-05 scientific contract identity mismatch")
    s=cfg["source"]
    if not (
        s["package"]=="scikit-image" and s["version"]=="0.25.2"
        and s["function"]=="skimage.data.camera"
        and s["photographer"]=="Lav Varshney" and s["license"]=="CC0"
        and s["raw_shape"]==[512,512] and s["raw_dtype"]=="uint8"
        and s["crop_rows"]==[128,384] and s["crop_cols"]==[128,384]
        and s["crop_shape"]==[256,256]
    ):
        raise ValueError("photo origin/shape/crop/rights contract changed")
    i=cfg["input"]
    if (
        i["intensity_scale_divisor"]!=255.0 or i["dtype"]!="float64"
        or i["numerical_clip"] is not False or
        i["noise"]!={"bit_generator":"PCG64","seed":20261009,
                     "normal_mean":0.0,"normal_std":0.08}
    ):
        raise ValueError("photo noise and normalization contract changed")
    p=cfg["solver"]
    if (
        p["N"]!=256 or p["h"]!=1.0 or p["dt"]!=0.2 or
        p["sigma"]!=1.5 or p["kappa"]!=0.075
        or p["methods"]!=list(METHODS) or p["gaussian_mode"]!="reflect"
        or p["truncate"]!=4.0 or p["sampling_t"]!=list(TIMES)
    ):
        raise ValueError("PDE or timing contract changed")
    lim=cfg["invariants"]
    if lim != {
        "max_mean_drift":1e-12,
        "max_range_violation":1e-12,
        "max_centered_ss_increase_per_step":1e-9,
        "max_flux_update_residual_per_step":1e-12
    }:
        raise ValueError("invariant limits were changed")
    if not all(math.isclose(round(t/p["dt"])*p["dt"],t,abs_tol=1e-12,rel_tol=0)
               for t in TIMES):
        raise ValueError("sampling times are not exactly representable in steps")
    return p["dt"],p["sigma"],p["kappa"],int(round(TIMES[-1]/p["dt"]))


def load_camera(cfg:dict,provenance_file:Path,fixture_dir:Path)->tuple[np.ndarray,dict]:
    provenance=json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance["experiment_id"]!="ET004-05" or provenance["license_gate"]!="PASS":
        raise PermissionError("image rights or pinned package provenance did not pass")
    source=provenance["source"]
    if (
        source["package"]!="scikit-image" or source["version"]!="0.25.2"
        or source["license"]!="CC0" or source["photographer"]!="Lav Varshney"
        or source["license_evidence_url"]!=cfg["source"]["doc_url"]
    ):
        raise PermissionError("provenance is not the preregistered CC0 image")
    if provenance["config_sha256"]!=sha_file(ROOT/"config"/"ET004-05.json"):
        raise AssertionError("config changed after image preparation")
    original_file=fixture_dir/Path(cfg["source"]["original_png"]).name
    cropped_file=fixture_dir/Path(cfg["source"]["cropped_png"]).name
    if not (original_file.is_file() and cropped_file.is_file()):
        raise FileNotFoundError("camera fixture missing")
    if sha_file(original_file)!=provenance["original"]["raw_png_sha256"]:
        raise AssertionError("original camera PNG was altered")
    if sha_file(cropped_file)!=provenance["crop"]["png_sha256"]:
        raise AssertionError("cropped camera PNG was altered")
    with Image.open(original_file) as f:
        original=np.asarray(f).copy()
    with Image.open(cropped_file) as f:
        crop=np.asarray(f).copy()
    if original.dtype!=np.uint8 or original.shape!=(512,512):
        raise AssertionError("incorrect original photograph pixels")
    if crop.dtype!=np.uint8 or crop.shape!=(256,256):
        raise AssertionError("incorrect cropped photograph pixels")
    if not np.array_equal(original[128:384,128:384],crop):
        raise AssertionError("cropped photograph is not the exact registered ROI")
    if hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()!=provenance["original"]["uint8_pixel_sha256"]:
        raise AssertionError("original camera pixels SHA256 invalid")
    if hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()!=provenance["crop"]["uint8_pixel_sha256"]:
        raise AssertionError("cropped camera pixels SHA256 invalid")
    clean=crop.astype(np.float64)/255.0
    if sha_float64(clean)!=provenance["crop"]["float64_div255_pixel_sha256"]:
        raise AssertionError("float64 normalized image SHA256 invalid")
    return clean,provenance


def measure(U:np.ndarray,B:np.ndarray)->dict:
    if U.shape!=(256,256) or B.shape!=(256,256) or not (np.isfinite(U).all() and np.isfinite(B).all()):
        raise FloatingPointError("invalid image for photo metrics")
    delta=U-B
    rmse=float(np.sqrt(np.mean(delta*delta)))
    mean=float(U.mean())
    return {
        "rmse":rmse,
        "psnr_db":math.inf if rmse==0 else 20*math.log10(1/rmse),
        "mean":mean,"minimum":float(U.min()),"maximum":float(U.max()),
        "centered_ss":float(np.sum((U-mean)**2))
    }


def audit(old:np.ndarray,next_u:np.ndarray,*,dt:float,sigma:float,kappa:float,
          method:str,initial_mean:float,initial_min:float,initial_max:float,
          old_ss:float,limits:dict)->dict:
    ax,ay=_face_coefficients(old,method=method,sigma=sigma,kappa=kappa,h=1.0)
    low_f=min(float(ax.min()),float(ay.min()))
    high_f=max(float(ax.max()),float(ay.max()))
    if not (0<low_f<=high_f<=1 and np.isfinite(ax).all() and np.isfinite(ay).all()):
        raise AssertionError("nonfinite or inadmissible shared-face diffusivity")
    d=np.zeros_like(old)
    x=dt*ax*(old[:,1:]-old[:,:-1])
    y=dt*ay*(old[1:,:]-old[:-1,:])
    d[:,:-1]+=x
    d[:,1:]-=x
    d[:-1,:]+=y
    d[1:,:]-=y
    flux_residual=float(np.max(np.abs((next_u-old)-d)))
    if not (np.isfinite(d).all() and np.isfinite(next_u).all()):
        raise FloatingPointError("nonfinite PDE update")
    mn,mx=float(next_u.min()),float(next_u.max())
    mean=float(next_u.mean())
    ss=float(np.sum((next_u-mean)**2))
    drift=abs(mean-initial_mean)
    exceed=max(0.0,initial_min-mn,mx-initial_max)
    increase=max(0.0,ss-old_ss)
    if (
        flux_residual>limits["max_flux_update_residual_per_step"] or
        drift>limits["max_mean_drift"] or
        exceed>limits["max_range_violation"] or
        increase>limits["max_centered_ss_increase_per_step"]
    ):
        raise AssertionError(
            f"PDE invariant failed {method}: "
            f"flux={flux_residual},drift={drift},range={exceed},variance={increase}"
        )
    return {
        "mean":mean,"mean_drift":drift,"minimum":mn,"maximum":mx,
        "range_violation":exceed,"centered_ss":ss,"centered_ss_increase":increase,
        "flux_update_residual":flux_residual,
        "minimum_face_diffusivity":low_f,"maximum_face_diffusivity":high_f
    }


def run(cfg_path:Path,fixture_dir:Path,provenance_file:Path,outdir:Path,source_sha:str)->None:
    if len(source_sha)!=40 or not all(ch in "0123456789abcdef" for ch in source_sha):
        raise ValueError("formal runner requires a complete source commit SHA")
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    dt,sigma,kappa,steps=require_parameters(cfg)
    if outdir.exists():
        raise FileExistsError("ET004-05 results already exist: must not overwrite")
    clean,provenance=load_camera(cfg,provenance_file,fixture_dir)
    rng=np.random.Generator(np.random.PCG64(20261009))
    noisy=clean+rng.normal(0.0,0.08,size=clean.shape)
    if noisy.dtype!=np.float64 or not np.isfinite(noisy).all():
        raise AssertionError("invalid initial noisy photograph")
    source_mean=float(noisy.mean())
    source_min,source_max=float(noisy.min()),float(noisy.max())
    initial_ss=float(np.sum((noisy-source_mean)**2))
    noisy_sha=sha_float64(noisy)
    print("ET004-05 FROZEN_IMAGE clean_float64_sha256="+sha_float64(clean)+
          " noisy_float64_sha256="+noisy_sha,flush=True)
    snapshots={"camera_clean":clean.copy(),"noisy":noisy.copy()}
    csv_rows=[{"method":"noisy","t":0,"step":0,**measure(noisy,clean)}]
    audit_rows=[]
    maxima={}
    clock_steps={int(round(t/dt)):t for t in TIMES if t>0}
    for method in METHODS:
        U=noisy.copy()
        prev_ss=initial_ss
        maxima[method]={
            "max_mean_drift":0.0,"max_range_violation":0.0,
            "max_centered_ss_increase":0.0,"max_flux_update_residual":0.0,
            "face_min":1.0,"face_max":0.0,"number_of_steps":steps
        }
        for step in range(1,steps+1):
            next_u=advance_one_step(U,dt=dt,h=1.0,method=method,sigma=sigma,kappa=kappa)
            stat=audit(
                U,next_u,dt=dt,sigma=sigma,kappa=kappa,method=method,
                initial_mean=source_mean,initial_min=source_min,initial_max=source_max,
                old_ss=prev_ss,limits=cfg["invariants"]
            )
            prev_ss=stat["centered_ss"]
            for key,other in (
                ("max_mean_drift","mean_drift"),
                ("max_range_violation","range_violation"),
                ("max_centered_ss_increase","centered_ss_increase"),
                ("max_flux_update_residual","flux_update_residual")
            ):
                maxima[method][key]=max(maxima[method][key],stat[other])
            maxima[method]["face_min"]=min(maxima[method]["face_min"],stat["minimum_face_diffusivity"])
            maxima[method]["face_max"]=max(maxima[method]["face_max"],stat["maximum_face_diffusivity"])
            audit_rows.append({"method":method,"step":step,"t":step*dt,**stat})
            U=next_u
            if step in clock_steps:
                t=clock_steps[step]
                snapshots[f"{method}_t{t:03d}"]=U.copy()
                row={"method":method,"t":t,"step":step,**measure(U,clean)}
                csv_rows.append(row)
                print(f"ET004-05 {method} t={t} RMSE={row['rmse']:.16e} "
                      f"PSNR={row['psnr_db']:.12f} mean={row['mean']:.16e}",flush=True)
    if len(snapshots)!=12 or len(csv_rows)!=11 or len(audit_rows)!=320:
        raise AssertionError("missing scheduled snapshots/metrics/step audits")
    outdir.mkdir(parents=True,exist_ok=False)
    def save_csv(name,headers,rows):
        with (outdir/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=headers,lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    save_csv("metrics.csv",METRIC_COLUMNS,csv_rows)
    save_csv("step_invariants.csv",STEP_COLUMNS,audit_rows)
    np.savez_compressed(outdir/"snapshots.npz",**snapshots)
    provenance["noisy"]={
        "bit_generator":"PCG64","seed":20261009,
        "normal_mean":0.0,"normal_std":0.08,
        "float64_pixel_sha256":noisy_sha,
        "intensity_clip_applied":False,
        "minimum":source_min,"maximum":source_max
    }
    provenance["pde_numeric_runtime"]={
        "python":sys.version.split()[0],"numpy":np.__version__,
        "scipy":metadata.version("scipy"),"pillow":metadata.version("Pillow"),
        "platform":platform.platform()
    }
    provenance["source_commit"]=source_sha
    (outdir/"source-provenance.json").write_text(
        json.dumps(provenance,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    hashes={
      "diffusion_solver.py":sha_file(ROOT/"src"/"diffusion_solver.py"),
      "run_ET004_05.py":sha_file(Path(__file__).resolve()),
      "prepare_ET004_05_camera.py":sha_file(ROOT/"src"/"prepare_ET004_05_camera.py"),
      "validate_ET004_05.py":sha_file(ROOT/"src"/"validate_ET004_05.py")
    }
    checks={
        "experiment_id":"ET004-05","status":"PASS",
        "scientific_scope":"real CC0 photograph with deliberately added Gaussian noise",
        "not_a_claim":"not a clean physical camera ground truth and not uncalibrated sensor noise",
        "source_commit":source_sha,"runner_type":"GitHub self-hosted",
        "config":cfg,"config_sha256":sha_file(cfg_path),
        "source_sha256":hashes,
        "runtime":provenance["pde_numeric_runtime"],
        "input":{
            "photo_source_provenance":"source-provenance.json",
            "original_png_sha256":provenance["original"]["raw_png_sha256"],
            "original_uint8_pixels_sha256":provenance["original"]["uint8_pixel_sha256"],
            "crop_png_sha256":provenance["crop"]["png_sha256"],
            "crop_float64_sha256":sha_float64(clean),
            "noisy_float64_sha256":noisy_sha,
            "initial_mean":source_mean,"initial_min":source_min,
            "initial_max":source_max,"initial_centered_ss":initial_ss
        },
        "outputs_sha256":{name:sha_file(outdir/name) for name in (
            "metrics.csv","step_invariants.csv","snapshots.npz","source-provenance.json"
        )},
        "invariants":maxima,
        "metric_rows":len(csv_rows),"step_rows":len(audit_rows),
        "snapshot_arrays":len(snapshots),"source_license_gate":"PASS",
        "not_verified":["unknown sensor noise restored","photographic scene physical truth",
                        "arbitrary-image optimality","FT004 formally rendered"]
    }
    (outdir/"checks.json").write_text(
        json.dumps(checks,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("ET004-05 FORMAL_GATE PASS "
          f"metrics=11 snapshots=12 steps=320 "
          f"ORIGINAL_PIXELS_SHA256={provenance['original']['uint8_pixel_sha256']} "
          f"NPZ_SHA256={checks['outputs_sha256']['snapshots.npz']}",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True,type=Path)
    ap.add_argument("--fixture-dir",required=True,type=Path)
    ap.add_argument("--provenance",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--source-sha",default=os.getenv("GITHUB_SHA",""))
    a=ap.parse_args()
    run(a.config,a.fixture_dir,a.provenance,a.out,a.source_sha)
