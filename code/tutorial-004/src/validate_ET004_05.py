"""ET004-05 independent read-only validation (no PDE calls, no skimage import).

Reopens original and cropped PNGs, independently rebuilds the fixed noise,
recomputes all photo metrics from stored float64 snapshots, and checks all
two-method step diagnostics and source/provenance SHA256 contracts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

METHODS = ("heat","regularized")
TIMES = (0,2,4,8,16,32)
METRICS = ("method","t","step","rmse","psnr_db","mean","minimum","maximum","centered_ss")
STEP = ("method","step","t","mean","mean_drift","minimum","maximum",
        "range_violation","centered_ss","centered_ss_increase","flux_update_residual",
        "minimum_face_diffusivity","maximum_face_diffusivity")


def file_hash(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def array_hash(a:np.ndarray)->str:
    if a.dtype!=np.float64 or a.shape!=(256,256):
        raise TypeError("invalid float64 camera image array")
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()


def assert_ok(ok:bool,message:str)->None:
    if not ok:
        raise AssertionError("ET004-05 independent validator FAIL: "+message)


def near(a,b,message,atol=1e-12):
    assert_ok(
        math.isfinite(float(a)) and math.isfinite(float(b)) and
        math.isclose(float(a),float(b),rel_tol=5e-11,abs_tol=atol),
        message
    )


def raw_metrics(U:np.ndarray,B:np.ndarray)->dict:
    err=U-B
    rmse=float(np.sqrt(np.mean(err**2)))
    mean=float(np.mean(U))
    return {
      "rmse":rmse,
      "psnr_db":math.inf if rmse==0 else 20*math.log10(1/rmse),
      "mean":mean,"minimum":float(U.min()),"maximum":float(U.max()),
      "centered_ss":float(np.sum((U-mean)**2))
    }


def validate(out:Path,cfg_path:Path,fixture:Path,src:Path,source_sha:str)->dict:
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    checks=json.loads((out/"checks.json").read_text(encoding="utf-8"))
    provenance=json.loads((out/"source-provenance.json").read_text(encoding="utf-8"))
    assert_ok(checks["status"]=="PASS" and checks["experiment_id"]=="ET004-05","formal scientific gate")
    assert_ok(checks["runner_type"]=="GitHub self-hosted" and len(source_sha)==40,
              "valid private Runner commit")
    assert_ok(checks["source_commit"]==source_sha==provenance["source_commit"],"source commit")
    assert_ok(checks["source_license_gate"]==provenance["license_gate"]=="PASS","CC0 source gate")
    assert_ok(checks["config"]==cfg and checks["config_sha256"]==file_hash(cfg_path),
              "frozen config and hash")
    assert_ok(provenance["config_sha256"]==file_hash(cfg_path),"package extraction config")
    assert_ok(cfg["experiment_id"]=="ET004-05" and cfg["status"]=="preregistered_before_formal_execution",
              "correct preregistration")
    assert_ok(cfg["scientific_contract"]=="notes/tutorial-004/experiment-spec.md","scientific contract path")
    assert_ok(
        cfg["source"]["version"]=="0.25.2" and cfg["source"]["package"]=="scikit-image" and
        cfg["source"]["function"]=="skimage.data.camera" and
        cfg["source"]["license"]=="CC0" and cfg["source"]["photographer"]=="Lav Varshney" and
        cfg["source"]["raw_shape"]==[512,512] and cfg["source"]["raw_dtype"]=="uint8" and
        cfg["source"]["crop_rows"]==[128,384] and cfg["source"]["crop_cols"]==[128,384],
        "pinned source and region"
    )
    ps=provenance["source"]
    assert_ok(
      ps["package"]=="scikit-image" and ps["version"]=="0.25.2" and
      ps["photographer"]=="Lav Varshney" and ps["license"]=="CC0" and
      ps["license_evidence_url"]=="https://scikit-image.org/docs/0.25.x/api/skimage.data.html",
      "pinned documentation and original photographer"
    )
    assert_ok(len(ps["camera_function_docstring_sha256"])==64,"license docs fingerprint")
    assert_ok(cfg["solver"]=={
      "N":256,"h":1.0,"dt":0.2,"sigma":1.5,"kappa":0.075,
      "methods":["heat","regularized"],"gaussian_mode":"reflect",
      "truncate":4.0,"sampling_t":[0,2,4,8,16,32]
    },"frozen nonlinear/linear solver contract")
    assert_ok(cfg["input"]=={
      "intensity_scale_divisor":255.0,"dtype":"float64",
      "noise":{"bit_generator":"PCG64","seed":20261009,
               "normal_mean":0.0,"normal_std":0.08},
      "numerical_clip":False
    },"fixed added Gaussian noise and clip policy")
    assert_ok(cfg["invariants"]=={
      "max_mean_drift":1e-12,"max_range_violation":1e-12,
      "max_centered_ss_increase_per_step":1e-9,
      "max_flux_update_residual_per_step":1e-12
    },"invariant constraints")
    assert_ok(checks["runtime"]["python"].startswith("3.12."),"Python 3.12")
    for name in ("diffusion_solver.py","prepare_ET004_05_camera.py",
                 "run_ET004_05.py","validate_ET004_05.py"):
        assert_ok(checks["source_sha256"][name]==file_hash(src/name),"source identity "+name)
    for name in ("metrics.csv","snapshots.npz","step_invariants.csv","source-provenance.json"):
        assert_ok(checks["outputs_sha256"][name]==file_hash(out/name),"result file SHA256 "+name)

    raw_png=fixture/"camera_original_cc0_512.png"
    cropped_png=fixture/"camera_clean_cc0.png"
    assert_ok(file_hash(raw_png)==provenance["original"]["raw_png_sha256"],
              "exact raw package camera PNG")
    assert_ok(file_hash(cropped_png)==provenance["crop"]["png_sha256"],
              "exact cropped camera PNG")
    with Image.open(raw_png) as img:
        original=np.asarray(img).copy()
    with Image.open(cropped_png) as img:
        cropped=np.asarray(img).copy()
    assert_ok(original.shape==(512,512) and original.dtype==np.uint8,
              "full original image dimensions/dtype")
    assert_ok(cropped.shape==(256,256) and cropped.dtype==np.uint8,
              "cropped image dimensions/dtype")
    assert_ok(np.array_equal(original[128:384,128:384],cropped),
              "fixed integer crop pixel-for-pixel")
    assert_ok(hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()==
              provenance["original"]["uint8_pixel_sha256"],
              "full source uint8 pixel SHA256")
    assert_ok(hashlib.sha256(np.ascontiguousarray(cropped).tobytes()).hexdigest()==
              provenance["crop"]["uint8_pixel_sha256"],
              "crop uint8 pixel SHA256")
    clean=cropped.astype(np.float64)/255.0
    assert_ok(array_hash(clean)==provenance["crop"]["float64_div255_pixel_sha256"],
              "clean float64 normalized array SHA256")
    assert_ok(array_hash(clean)==checks["input"]["crop_float64_sha256"],
              "clean input SHA256")
    # Reconstruct registered synthetic noise independently without importing
    # the experiment input module or using the solver.
    generator=np.random.Generator(np.random.PCG64(20261009))
    noisy=clean+generator.normal(0.0,0.08,size=(256,256))
    assert_ok(noisy.dtype==np.float64 and np.isfinite(noisy).all(),
              "valid reconstructed noise")
    assert_ok(array_hash(noisy)==provenance["noisy"]["float64_pixel_sha256"]==
              checks["input"]["noisy_float64_sha256"],
              "replayed PCG64 noise SHA256 and no clipping")
    assert_ok(not np.array_equal(noisy,np.clip(noisy,0,1)) or
              provenance["noisy"]["intensity_clip_applied"] is False,
              "no hidden numeric clipping")

    expected_keys={"camera_clean","noisy"} | {
        f"{m}_t{t:03d}" for m in METHODS for t in TIMES if t>0}
    with np.load(out/"snapshots.npz",allow_pickle=False) as archive:
        assert_ok(set(archive.files)==expected_keys,"12 complete float64 snapshots")
        snapshots={name:archive[name].copy() for name in archive.files}
    for name,a in snapshots.items():
        assert_ok(a.shape==(256,256) and a.dtype==np.float64 and np.isfinite(a).all(),
                  "snapshot shape/type/finite "+name)
    assert_ok(np.array_equal(snapshots["camera_clean"],clean),"exact clean snapshot")
    assert_ok(np.array_equal(snapshots["noisy"],noisy),"exact noisy snapshot")

    with (out/"metrics.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert_ok(tuple(reader.fieldnames or [])==METRICS,"metric CSV stable schema")
        metric_rows=list(reader)
    expected_rows=[("noisy",0)]+[(m,t) for m in METHODS for t in TIMES if t>0]
    assert_ok(len(metric_rows)==11==checks["metric_rows"],"11 checkpoint rows")
    initial_mean=float(noisy.mean())
    initial_low,initial_high=float(noisy.min()),float(noisy.max())
    initial_var=float(np.sum((noisy-initial_mean)**2))
    for row,(method,t) in zip(metric_rows,expected_rows):
        assert_ok(row["method"]==method and int(row["t"])==t,
                  f"exact method/time row {method}/{t}")
        assert_ok(int(row["step"])==round(t/0.2),"time step count")
        u=noisy if method=="noisy" else snapshots[f"{method}_t{t:03d}"]
        for key,val in raw_metrics(u,clean).items():
            near(row[key],val,f"metric {method} t={t} {key}")
        assert_ok(abs(float(row["mean"])-initial_mean)<=cfg["invariants"]["max_mean_drift"],
                  "sampled mean conservation")
        assert_ok(float(row["minimum"])>=initial_low-1e-12 and
                  float(row["maximum"])<=initial_high+1e-12,
                  "sampled maximum principle")

    with (out/"step_invariants.csv").open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        assert_ok(tuple(reader.fieldnames or [])==STEP,"step diagnostics schema")
        step_rows=list(reader)
    assert_ok(len(step_rows)==320==checks["step_rows"],"320 total step audit records")
    assert_ok(checks["snapshot_arrays"]==12,"exact number of snapshots")
    idx=0
    for method in METHODS:
        rows=step_rows[idx:idx+160]; idx+=160
        prior_ss=initial_var
        computed={
            "max_mean_drift":0.0,"max_range_violation":0.0,
            "max_centered_ss_increase":0.0,"max_flux_update_residual":0.0,
            "face_min":1.0,"face_max":0.0
        }
        for step,row in enumerate(rows,1):
            assert_ok(row["method"]==method and int(row["step"])==step,
                      "method/time sequence")
            near(row["t"],step*0.2,"artificial time",atol=1e-11)
            stats={key:float(row[key]) for key in STEP if key not in ("method","step")}
            assert_ok(all(math.isfinite(v) for v in stats.values()),"finite step diagnostics")
            mean=stats["mean"]
            near(stats["mean_drift"],abs(mean-initial_mean),"mean-drift consistency")
            near(stats["range_violation"],max(0.0,initial_low-stats["minimum"],
                  stats["maximum"]-initial_high),"range consistency")
            near(stats["centered_ss_increase"],max(0.0,stats["centered_ss"]-prior_ss),
                 "variance-growth consistency",atol=1e-9)
            assert_ok(0<stats["minimum_face_diffusivity"]<=stats["maximum_face_diffusivity"]<=1,
                      "face diffusivity inside (0,1]")
            for key,limit in (
                ("mean_drift","max_mean_drift"),
                ("range_violation","max_range_violation"),
                ("centered_ss_increase","max_centered_ss_increase_per_step"),
                ("flux_update_residual","max_flux_update_residual_per_step")
            ):
                assert_ok(0<=stats[key]<=cfg["invariants"][limit],
                          "pre-registered step invariant "+key)
            computed["max_mean_drift"]=max(computed["max_mean_drift"],stats["mean_drift"])
            computed["max_range_violation"]=max(computed["max_range_violation"],stats["range_violation"])
            computed["max_centered_ss_increase"]=max(computed["max_centered_ss_increase"],stats["centered_ss_increase"])
            computed["max_flux_update_residual"]=max(computed["max_flux_update_residual"],stats["flux_update_residual"])
            computed["face_min"]=min(computed["face_min"],stats["minimum_face_diffusivity"])
            computed["face_max"]=max(computed["face_max"],stats["maximum_face_diffusivity"])
            if step in (10,20,40,80,160):
                target=snapshots[f"{method}_t{int(round(step*0.2)):03d}"]
                near(stats["mean"],float(target.mean()),"sampled actual mean")
                near(stats["minimum"],float(target.min()),"sampled actual min")
                near(stats["maximum"],float(target.max()),"sampled actual max")
            prior_ss=stats["centered_ss"]
        diag=checks["invariants"][method]
        assert_ok(diag["number_of_steps"]==160,"all steps for "+method)
        for key in computed:
            near(diag[key],computed[key],"per-method audit maxima "+key)
    assert_ok(checks["source_license_gate"]=="PASS","rights gate")
    return {
        "experiment_id":"ET004-05","status":"PASS",
        "verifier":"read-only; no PDE advancement or skimage import",
        "source_commit":source_sha,
        "license":"CC0",
        "photographer":"Lav Varshney",
        "camera_source_version":"scikit-image 0.25.2",
        "raw_png_sha256":provenance["original"]["raw_png_sha256"],
        "raw_uint8_pixel_sha256":provenance["original"]["uint8_pixel_sha256"],
        "crop_png_sha256":provenance["crop"]["png_sha256"],
        "crop_float64_sha256":array_hash(clean),
        "noisy_float64_sha256":array_hash(noisy),
        "metric_records":len(metric_rows),"step_records":len(step_rows),
        "snapshots_count":len(snapshots),
        "output_sha256":{
            name:file_hash(out/name) for name in (
                "metrics.csv","snapshots.npz","step_invariants.csv",
                "source-provenance.json","checks.json"
            )
        },
        "not_verified":["unobserved true scene","native camera noise","universal optimality","FT004 figures"]
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--fixture-dir",required=True,type=Path)
    p.add_argument("--src",required=True,type=Path)
    p.add_argument("--source-sha",required=True)
    a=p.parse_args()
    result=validate(a.results,a.config,a.fixture_dir,a.src,a.source_sha)
    (a.results/"validation.json").write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("ET004-05 INDEPENDENT_VALIDATOR PASS "
          "license=CC0 author=Lav_Varshney metrics=11 steps=320 "
          "ORIGINAL_PIXELS_SHA256="+result["raw_uint8_pixel_sha256"]+" "
          "NPZ_SHA256="+result["output_sha256"]["snapshots.npz"],flush=True)
