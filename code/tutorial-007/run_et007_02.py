#!/usr/bin/env python3
"""ET007-02: actually solve the 45-degree affine Dirichlet experiment.

Use the already-verified production matrix/flux model without touching ET007-01.
N=32/64 x M0/M1 are the four cases preregistered in design_v01.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from run_experiments import (SUMMARY as COMMON_SUMMARY, digest, write_csv,
                             write_json, save_fields)
from src.model import assemble, dual_face_flux, nodal_solution, report_metrics

ROOT = Path(__file__).resolve().parent
STAGE = "ET007-02"
EXTRA = ["q_x_mean", "q_y_mean", "q_x_true", "q_y_true"]
SUMMARY = COMMON_SUMMARY[:-1] + EXTRA + ["solver_status"]


def frozen_setup(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config["article_id"] != "tutorial-007"
            or config["design_version"] != "v0.2"
            or config["status"] != "science_parameters_frozen"
            or config["tensor_principal"] != [10.0, 1.0]
            or config["domain"] != [0.0, 1.0, 0.0, 1.0]
            or config["methods"] != ["M0", "M1"]
            or config["planned_unique_solve_count"] != 28):
        raise ValueError("input does not match the frozen Tutorial-007 science contract")
    stage = next(s for s in config["experiment_sets"] if s["id"] == STAGE)
    if (stage["field"] != "affine"
            or stage["theta_deg"] != 45.0
            or stage["grids_N"] != [32, 64]
            or stage["methods"] != ["M0", "M1"]
            or stage["save_full_fields_N"] != [64]):
        raise ValueError("ET007-02 case matrix was changed after freeze")
    return config, stage


def run(config_path: Path, output: Path) -> None:
    config, stage = frozen_setup(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    tol = config["tolerances"]
    principal = tuple(config["tensor_principal"])
    source_files = {
        "src/model.py": digest(ROOT/"src/model.py"),
        "run_experiments.py": digest(ROOT/"run_experiments.py"),
        "run_et007_02.py": digest(ROOT/"run_et007_02.py"),
    }
    records: list[dict] = []
    for n in stage["grids_N"]:
        for method in stage["methods"]:
            system = assemble(n, stage["theta_deg"], method, "affine", principal)
            u, residual = nodal_solution(system)  # A genuine sparse solve.
            qx, qy = dual_face_flux(u, 45.0, method, principal)
            metrics = report_metrics(u, 45.0, method, "affine", principal)
            actual_qx, actual_qy = float(np.mean(qx)), float(np.mean(qy))
            reference_qx, reference_qy = -5.5, -4.5
            case_id = f"affine-p45-N{n}-{method}"
            if not math.isfinite(residual) or residual > tol["linear_residual_linf_relative_max"]:
                raise AssertionError(f"linear solve residual not acceptable: {case_id}: {residual}")
            if (metrics["dual_cell_balance_scaled"] > tol["dual_cell_balance_scaled_max"]
                    or metrics["dual_boundary_budget_scaled"] > tol["dual_cell_balance_scaled_max"]):
                raise AssertionError(f"internal dual-control-volume budget failed: {case_id}")
            if metrics["T_error_linf"] > tol["affine_solution_linf_max"]:
                raise AssertionError(f"affine Dirichlet temperature failed: {case_id}")
            if metrics["q_x_error_linf"] > tol["affine_true_m1_flux_linf_max"]:
                raise AssertionError(f"true horizontal flux failed: {case_id}")
            if method == "M1":
                if metrics["q_y_error_linf"] > tol["affine_true_m1_flux_linf_max"]:
                    raise AssertionError(f"complete-tensor transverse flux failed: {case_id}")
            else:
                if abs(metrics["q_y_error_linf"] - 4.5) > tol["affine_true_m1_flux_linf_max"]:
                    raise AssertionError(f"missing M0 cross-flux counterexample failed: {case_id}")
                if abs(actual_qy) > tol["affine_true_m1_flux_linf_max"]:
                    raise AssertionError(f"M0 transverse flux should be zero: {case_id}")
            if not all(math.isfinite(v) for v in [actual_qx, actual_qy, *metrics.values()]):
                raise FloatingPointError(f"nonfinite scientific results: {case_id}")
            path = output/"raw"/case_id
            path.mkdir(parents=True, exist_ok=False)
            if n in stage["save_full_fields_N"]:
                save_fields(path, u, qx, qy)
            write_json(path/"run.json", {
                "article_id": "tutorial-007", "experiment": STAGE, "case_id": case_id,
                "field": "affine", "theta_deg": 45.0, "N": n, "method": method,
                "status": "solved",
                "configuration_sha256": digest(config_path),
                "source_sha256": source_files,
                "source_commit": os.environ.get("GITHUB_SHA"),
                "matrix_scaling": "unscaled_Lh_dirichlet_eliminated",
                "control_volume_scope": "Omega_h_not_physical_boundary",
                "scientific_csv_sha256": {
                    p.name: digest(p) for p in sorted(path.glob("*.csv"))
                },
            })
            row = {
                "case_id": case_id, "experiment": STAGE, "field": "affine",
                "theta_deg": 45.0, "N": n, "h": 1.0/n, "method": method,
                "K_xx": system.k[0], "K_xy": system.k[1], "K_yy": system.k[2],
                "n_unknowns": (n-1)**2, "nnz": system.A.nnz,
                "linear_residual_linf_relative": residual,
                **metrics,
                "q_x_mean": actual_qx, "q_y_mean": actual_qy,
                "q_x_true": reference_qx, "q_y_true": reference_qy,
                "solver_status": "solved",
            }
            records.append(row)
            print("ACTUAL_ET007_02_SOLVE", case_id,
                  "T_max_error", format(row["T_error_linf"], ".17g"),
                  "mean_qx", format(actual_qx, ".17g"),
                  "mean_qy", format(actual_qy, ".17g"),
                  "true_qy", reference_qy,
                  "linear_residual", format(residual, ".17g"), flush=True)
    if len(records) != 4 or len({r["case_id"] for r in records}) != 4:
        raise AssertionError("ET007-02 requires four unique actual sparse solves")
    write_csv(output/"summary.csv", SUMMARY, records)
    scientific = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output/"run.json", {
        "article_id": "tutorial-007", "experiment": STAGE,
        "status": "completed", "executed_case_count": 4,
        "planned_full_case_count": 28, "complete_28_case_program": False,
        "run_case_ids": [r["case_id"] for r in records],
        "config_sha256": digest(config_path),
        "source_sha256": source_files,
        "source_commit": os.environ.get("GITHUB_SHA"),
        "scientific_sha256": {
            str(p.relative_to(output)): digest(p) for p in scientific
        },
    })
    print("ET007_02_SCIENCE_OUTPUT_COMPLETE", len(records),
          "cases", len([p for p in output.rglob("*.csv")]), "scientific CSV", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=[STAGE])
    args = parser.parse_args()
    run(args.config.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
