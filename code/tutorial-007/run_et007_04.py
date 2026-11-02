#!/usr/bin/env python3
"""ET007-04: solve ONLY the eight pre-registered 15/30-degree cases.

Reuse, never rerun, the frozen 0-degree ET007-01 and 45-degree ET007-03
N=64/128 rows. The stage output includes an eight-row NEW summary and
a sixteen-row angle comparison with an explicit immutable source per row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from run_experiments import SUMMARY, digest, write_csv, write_json
from src.model import assemble, nodal_solution, report_metrics

ROOT = Path(__file__).resolve().parent
STAGE = "ET007-04"
# Scientific identities, captured from the already successful private runs.
FROZEN_SUMMARY_SHA256 = {
    "ET007-01": "1788ec0ae48ead3903d6d09cb1bf8b999aa31b42d5f37728061fb2eefbd88233",
    "ET007-02": "bec0f2916cfecdab732eb4d48d81c8a9cd814353e337db7ea4384dad41b543d8",
    "ET007-03": "df816b6eb9ff4de8c9d1e8d182a1279bb3926448dbb399f81f9d1d53d44d9fa2",
}
ANGULAR_FIELDS = SUMMARY + ["source_stage", "source_summary_sha256"]
ALL_ANGLES = (0, 15, 30, 45)
NEW_ANGLES = (15, 30)
GRID = (64, 128)
METHODS = ("M0", "M1")


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_setup(config_path: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["article_id"] != "tutorial-007"
            or config["design_version"] != "v0.2"
            or config["status"] != "science_parameters_frozen"
            or config["domain"] != [0, 1, 0, 1]
            or config["tensor_principal"] != [10, 1]
            or config["methods"] != list(METHODS)
            or config["planned_unique_solve_count"] != 28
            or config["theta_unit"] != "degrees"
            or config["matrix_scaling"] != "unscaled_Lh_operator"):
        raise ValueError("frozen Tutorial-007 mathematical design changed")
    stage = next(s for s in config["experiment_sets"] if s["id"] == STAGE)
    if (stage["field"] != "sinesine"
            or stage["theta_deg_values"] != list(ALL_ANGLES)
            or stage["new_theta_deg_values"] != list(NEW_ANGLES)
            or stage["grids_N"] != list(GRID)
            or stage["methods"] != list(METHODS)
            or stage["reuse_case_ids_from"] != ["ET007-01","ET007-03"]
            or stage["save_full_fields_N"] != []):
        raise ValueError("ET007-04 stage departed from the frozen case list")
    return config, stage


def require_frozen_summary(path: Path, stage: str) -> list[dict[str,str]]:
    expected = FROZEN_SUMMARY_SHA256[stage]
    actual = digest(path)
    if actual != expected:
        raise AssertionError(f"frozen {stage} summary byte identity mismatch")
    records = read_csv(path)
    count = {"ET007-01":8,"ET007-02":4,"ET007-03":8}[stage]
    if len(records) != count or len(set(r["case_id"] for r in records)) != count:
        raise AssertionError(f"incomplete/duplicate prior results: {stage}")
    if any(r["experiment"]!=stage or r["solver_status"]!="solved" for r in records):
        raise AssertionError(f"prior source is not an accepted science run: {stage}")
    return records


def angle_rows(frozen_0: list[dict[str,str]],
               fresh_15_30: list[dict],
               frozen_45: list[dict[str,str]]) -> list[dict]:
    # Reuse exact textual values from the prior source CSV: do not recompute
    # or reformat their measured science. Fresh rows retain 17g serialization.
    def serialize_fresh(r: dict) -> dict[str,str]:
        def val(v):
            if isinstance(v, (float,np.floating)):
                return format(float(v),".17g")
            return str(v)
        return {k:val(r[k]) for k in SUMMARY}
    source = {}
    for stage,rows in (("ET007-01",frozen_0),("ET007-03",frozen_45)):
        for r in rows:
            if int(r["N"]) in GRID and r["field"]=="sinesine":
                key=r["case_id"]
                if key in source:
                    raise AssertionError("duplicate frozen source row "+key)
                source[key]=(r,stage)
    for r in fresh_15_30:
        key=r["case_id"]
        if key in source:
            raise AssertionError("fresh result collides with frozen case")
        source[key]=(serialize_fresh(r),STAGE)
    joined=[]
    for deg in ALL_ANGLES:
        for n in GRID:
            for method in METHODS:
                key=f"sinesine-p{deg:02d}-N{n}-{method}"
                if key not in source:
                    raise AssertionError("missing angular comparison "+key)
                r,stage=source[key]
                if (r["case_id"]!=key or int(r["N"])!=n
                        or float(r["theta_deg"])!=deg or r["method"]!=method
                        or r["field"]!="sinesine"):
                    raise AssertionError("incorrect source row assigned to "+key)
                expected_fields=set(SUMMARY)
                if set(r)!=expected_fields:
                    raise AssertionError("unrecognized science schema in "+key)
                joined.append({**r,"source_stage":stage,
                               "source_summary_sha256":
                                   FROZEN_SUMMARY_SHA256[stage] if stage!=STAGE else
                                   "CURRENT_STAGE_SUMMARY_PENDING"})
    if len(joined)!=16 or len(source)!=16:
        raise AssertionError("angular scan must contain 16 unique points")
    return joined


def run(config_path: Path, output: Path, et01: Path, et02: Path, et03: Path) -> None:
    config,stage=frozen_setup(config_path)
    frozen0=require_frozen_summary(et01,"ET007-01")
    require_frozen_summary(et02,"ET007-02")  # Identity only; affine not used.
    frozen45=require_frozen_summary(et03,"ET007-03")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite results: {output}")
    output.mkdir(parents=True,exist_ok=False)
    principal=tuple(config["tensor_principal"])
    tolerance=config["tolerances"]
    sources={
        "src/model.py":digest(ROOT/"src/model.py"),
        "run_experiments.py":digest(ROOT/"run_experiments.py"),
        "run_et007_04.py":digest(ROOT/"run_et007_04.py"),
    }
    fresh=[]
    for angle in stage["new_theta_deg_values"]:
        for n in stage["grids_N"]:
            source_rhs={}
            for method in stage["methods"]:
                case=f"sinesine-p{angle:02d}-N{n}-{method}"
                sys=assemble(n,float(angle),method,"sinesine",principal)
                source_rhs[method]=sys.b.copy()
                if method=="M1" and not np.array_equal(source_rhs["M0"],sys.b):
                    raise AssertionError(f"{case}: methods use different physical right sides")
                # Sparse matrix actually constructed and solved, never replace
                # by the independent reference solutions.
                u,residual=nodal_solution(sys)
                if not math.isfinite(residual) or residual>tolerance["linear_residual_linf_relative_max"]:
                    raise AssertionError(f"linear residual exceeds frozen limit {case}: {residual}")
                metrics=report_metrics(u,float(angle),method,"sinesine",principal)
                if not all(math.isfinite(float(x)) for x in metrics.values()):
                    raise FloatingPointError(f"nonfinite temperature/flux/budget in {case}")
                if (metrics["dual_cell_balance_scaled"]>tolerance["dual_cell_balance_scaled_max"]
                        or metrics["dual_boundary_budget_scaled"]>tolerance["dual_cell_balance_scaled_max"]):
                    raise AssertionError(f"{case}: interior dual flux budget failed")
                raw=output/"raw"/case
                raw.mkdir(parents=True,exist_ok=False)
                # Frozen design deliberately prescribes NO full fields for ET04.
                write_json(raw/"run.json",{
                    "article_id":"tutorial-007","experiment":STAGE,
                    "case_id":case,"field":"sinesine","theta_deg":angle,
                    "N":n,"method":method,"status":"solved",
                    "complete_physical_source_for_both_methods":True,
                    "matrix_scaling":"unscaled_Lh_Dirichlet_eliminated",
                    "dual_control_region":"Omega_h_not_physical_boundary",
                    "config_sha256":digest(config_path),
                    "source_sha256":sources,
                    "source_commit":os.environ.get("GITHUB_SHA"),
                    "scientific_csv_sha256":{},
                })
                kxx,kxy,kyy=sys.k
                row={
                    "case_id":case,"experiment":STAGE,"field":"sinesine",
                    "theta_deg":angle,"N":n,"h":1.0/n,"method":method,
                    "K_xx":kxx,"K_xy":kxy,"K_yy":kyy,
                    "n_unknowns":(n-1)**2,"nnz":sys.A.nnz,
                    "linear_residual_linf_relative":residual,
                    **metrics,"solver_status":"solved",
                }
                fresh.append(row)
                print("ET007_04_ACTUAL_SPARSE_SOLVE",case,
                      "T_L2",format(metrics["T_error_l2"],".17g"),
                      "q_x_error",format(metrics["q_x_error_linf"],".17g"),
                      "q_y_error",format(metrics["q_y_error_linf"],".17g"),
                      "relative_residual",format(residual,".17g"),flush=True)
    if len(fresh)!=8 or len({r["case_id"] for r in fresh})!=8:
        raise AssertionError("must run exactly the eight new ET007-04 cases")
    write_csv(output/"summary.csv",SUMMARY,fresh)
    stage_summary_sha=digest(output/"summary.csv")
    view=angle_rows(frozen0,fresh,frozen45)
    for row in view:
        if row["source_stage"]==STAGE:
            row["source_summary_sha256"]=stage_summary_sha
    write_csv(output/"angle_comparison.csv",ANGULAR_FIELDS,view)
    files=sorted(p for p in output.rglob("*") if p.is_file())
    expected={"summary.csv","angle_comparison.csv",*(
        "raw/"+r["case_id"]+"/run.json" for r in fresh)}
    if {str(p.relative_to(output)) for p in files}!=expected:
        raise AssertionError("missing or extra scientific outputs")
    write_json(output/"run.json",{
        "article_id":"tutorial-007","experiment":STAGE,"status":"completed",
        "source_commit":os.environ.get("GITHUB_SHA"),
        "executed_case_count":8,
        "reused_case_count":8,
        "angle_comparison_count":16,
        "planned_full_case_count":28,
        "earlier_accepted_case_count":20,
        "complete_28_case_program":False,
        "stage_only_new_angles":[15,30],
        "raw_fields_saved":False,
        "new_case_ids":[r["case_id"] for r in fresh],
        "reused_case_ids":[r["case_id"] for r in view if r["source_stage"]!=STAGE],
        "frozen_summary_sha256":FROZEN_SUMMARY_SHA256,
        "stage_summary_sha256":stage_summary_sha,
        "config_sha256":digest(config_path),
        "source_sha256":sources,
        "scientific_sha256":{
            str(p.relative_to(output)):digest(p) for p in files
        },
    })
    print("ET007_04_ACTUAL_NEW_SOLVES_COMPLETE",len(fresh),
          "NEW; REUSED",8,"FROZEN; ANGULAR_POINTS",len(view),flush=True)


def main() -> None:
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--config",type=Path,required=True)
    arg.add_argument("--stage",choices=[STAGE],required=True)
    arg.add_argument("--et01-summary",type=Path,required=True)
    arg.add_argument("--et02-summary",type=Path,required=True)
    arg.add_argument("--et03-summary",type=Path,required=True)
    arg.add_argument("--out",type=Path,required=True)
    args=arg.parse_args()
    run(args.config.resolve(),args.out.resolve(),
        args.et01_summary.resolve(),args.et02_summary.resolve(),
        args.et03_summary.resolve())


if __name__=="__main__":
    main()
