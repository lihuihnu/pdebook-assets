#!/usr/bin/env python3
"""Run only the frozen aligned-tensor ET007-01 cases; never overwrite output.

This script performs sparse solves, not analytic substitution. The closed-form
discrete sine solution belongs solely to the independent validator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from src.model import assemble, nodal_solution, report_metrics, dual_face_flux

ROOT = Path(__file__).resolve().parent
STAGE = "ET007-01"
SUMMARY = [
    "case_id", "experiment", "field", "theta_deg", "N", "h", "method",
    "K_xx", "K_xy", "K_yy", "n_unknowns", "nnz",
    "linear_residual_linf_relative", "T_error_l2", "T_error_linf",
    "q_x_error_linf", "q_y_error_linf",
    "dual_cell_balance_linf", "dual_cell_balance_scaled",
    "dual_boundary_budget_defect", "dual_boundary_budget_scaled",
    "solver_status",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def format_scalar(value):
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({key: format_scalar(row[key]) for key in fields})


def frozen_setup(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config["article_id"] != "tutorial-007" or config["status"] != "science_parameters_frozen"
            or config["design_version"] != "v0.2"
            or config["tensor_principal"] != [10.0, 1.0]
            or config["domain"] != [0.0, 1.0, 0.0, 1.0]
            or config["methods"] != ["M0", "M1"]
            or config["planned_unique_solve_count"] != 28):
        raise ValueError("configuration does not match frozen scientific design")
    stage = next(s for s in config["experiment_sets"] if s["id"] == STAGE)
    if (stage["theta_deg"] != 0.0 or stage["field"] != "sinesine"
            or stage["grids_N"] != [16, 32, 64, 128]
            or stage["methods"] != ["M0", "M1"]
            or stage["save_full_fields_N"] != [64]):
        raise ValueError("ET007-01 stage deviates from frozen design")
    return config, stage


def save_fields(path: Path, u: np.ndarray, qx: np.ndarray, qy: np.ndarray):
    n = u.shape[0] - 1
    h = 1.0 / n
    write_csv(path/"temperature.csv", ["i", "j", "x", "y", "u"], [
        {"i": i, "j": j, "x": i*h, "y": j*h, "u": u[i,j]}
        for i in range(n+1) for j in range(n+1)])
    write_csv(path/"flux_x.csv", ["i", "j", "x", "y", "q_x"], [
        {"i": i, "j": j, "x": (i+0.5)*h, "y": j*h, "q_x": qx[i,j-1]}
        for i in range(n) for j in range(1,n)])
    write_csv(path/"flux_y.csv", ["i", "j", "x", "y", "q_y"], [
        {"i": i, "j": j, "x": i*h, "y": (j+0.5)*h, "q_y": qy[i-1,j]}
        for i in range(1,n) for j in range(n)])


def run(config_path: Path, output: Path) -> None:
    config, stage = frozen_setup(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    t = config["tolerances"]
    principal = tuple(config["tensor_principal"])
    rows = []
    source_files = {"src/model.py": digest(ROOT/"src/model.py"),
                    "run_experiments.py": digest(ROOT/"run_experiments.py")}
    for n in stage["grids_N"]:
        pair = {}
        for method in stage["methods"]:
            system = assemble(n, 0.0, method, "sinesine", principal)
            u, residual = nodal_solution(system)
            if residual > t["linear_residual_linf_relative_max"]:
                raise AssertionError(f"linear residual failed: N={n} method={method} value={residual}")
            metrics = report_metrics(u, 0.0, method, "sinesine", principal)
            scale = t["dual_cell_balance_scaled_max"]
            if (metrics["dual_cell_balance_scaled"] > scale
                    or metrics["dual_boundary_budget_scaled"] > scale):
                raise AssertionError(f"dual-control-volume balance failed: N={n}, method={method}")
            case_id = f"sinesine-p00-N{n}-{method}"
            row = {
                "case_id": case_id, "experiment": STAGE, "field": "sinesine",
                "theta_deg": 0., "N": n, "h": 1.0/n, "method": method,
                "K_xx": system.k[0], "K_xy": system.k[1], "K_yy": system.k[2],
                "n_unknowns": (n-1)**2, "nnz": system.A.nnz,
                "linear_residual_linf_relative": residual,
                **metrics, "solver_status": "solved",
            }
            if any(not np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("non-finite scientific diagnostic")
            raw = output/"raw"/case_id
            raw.mkdir(parents=True, exist_ok=False)
            if n in stage["save_full_fields_N"]:
                qx, qy = dual_face_flux(u, 0.0, method, principal)
                save_fields(raw, u, qx, qy)
            write_json(raw/"run.json", {
                "article_id": "tutorial-007", "experiment": STAGE,
                "case_id": case_id, "status": "solved",
                "field": "sinesine", "theta_deg": 0.0, "N": n, "method": method,
                "matrix_kind": "interior_unscaled_Lh_dirichlet_eliminated",
                "dual_control_region": "Omega_h",
                "source_commit": os.environ.get("GITHUB_SHA"),
                "config_sha256": digest(config_path), "source_sha256": source_files,
                "scientific_csv_sha256": {
                    p.name: digest(p) for p in sorted(raw.glob("*.csv"))
                },
            })
            rows.append(row)
            pair[method] = (system.A, system.b, u)
            print("SOLVED", case_id, "n_unknowns", row["n_unknowns"],
                  "T_L2", format_scalar(row["T_error_l2"]),
                  "residual", format_scalar(residual), flush=True)
        ma, ra, ua = pair["M0"]
        mb, rb, ub = pair["M1"]
        a_delta = ma - mb
        diff = max(float(np.max(np.abs(a_delta.data))) if a_delta.nnz else 0.,
                   float(np.max(np.abs(ra-rb))),
                   float(np.max(np.abs(ua-ub))))
        if diff > t["m0_m1_solution_linf_difference_max"]:
            raise AssertionError(f"M0/M1 unexpected disagreement at N={n}: {diff}")
    if len(rows) != 8:
        raise AssertionError("ET007-01 must have exactly eight unique solves")
    write_csv(output/"summary.csv", SUMMARY, rows)
    outputs = [p for p in output.rglob("*") if p.is_file()]
    write_json(output/"run.json", {
        "article_id": "tutorial-007", "experiment": STAGE,
        "status": "completed", "complete_28_case_program": False,
        "run_case_ids": [row["case_id"] for row in rows],
        "planned_full_case_count": 28, "executed_case_count": 8,
        "config_sha256": digest(config_path), "source_sha256": source_files,
        "source_commit": os.environ.get("GITHUB_SHA"),
        "artifact_sha256": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(outputs) if p.name != "run.json"
        },
    })
    print("ET007_01_PRODUCTION_COMPLETED", len(rows), "rows",
          sum(1 for p in output.rglob("*.csv")), "CSV files", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=[STAGE], required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
