#!/usr/bin/env python3
"""Run the frozen ET007-03 rotated sine-manufactured-source experiment.

Eight genuine sparse solves: theta=45 degrees, N=16/32/64/128, M0/M1.
M0 intentionally omits the cross diffusion term; BOTH methods consume
the same exact FULL-tensor source. Never recompute frozen ET007-01/02.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from run_experiments import SUMMARY, digest, write_csv, write_json, save_fields
from src.model import assemble, dual_face_flux, nodal_solution, report_metrics

ROOT = Path(__file__).resolve().parent
STAGE = "ET007-03"


def frozen_setup(config_path: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["article_id"] != "tutorial-007"
            or config["status"] != "science_parameters_frozen"
            or config["design_version"] != "v0.2"
            or config["tensor_principal"] != [10.0, 1.0]
            or config["domain"] != [0.0, 1.0, 0.0, 1.0]
            or config["methods"] != ["M0", "M1"]
            or config["planned_unique_solve_count"] != 28):
        raise ValueError("ET007-03 would depart from the frozen article contract")
    stage = next(s for s in config["experiment_sets"] if s["id"] == STAGE)
    if (stage["field"] != "sinesine"
            or stage["theta_deg"] != 45.0
            or stage["grids_N"] != [16, 32, 64, 128]
            or stage["methods"] != ["M0", "M1"]
            or stage["save_full_fields_N"] != [64, 128]):
        raise ValueError("ET007-03 stage differs from the preregistered matrix")
    if config["theta_unit"] != "degrees" or config["matrix_scaling"] != "unscaled_Lh_operator":
        raise ValueError("changed angle or matrix normalization")
    return config, stage


def run(config_path: Path, output: Path) -> None:
    config, stage = frozen_setup(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    tolerances = config["tolerances"]
    principal = tuple(config["tensor_principal"])
    source_files = {
        "src/model.py": digest(ROOT / "src/model.py"),
        "run_experiments.py": digest(ROOT / "run_experiments.py"),
        "run_et007_03.py": digest(ROOT / "run_et007_03.py"),
    }
    recorded: list[dict] = []
    for n in stage["grids_N"]:
        pair_sources: dict[str, np.ndarray] = {}
        for method in stage["methods"]:
            case_id = f"sinesine-p45-N{n}-{method}"
            system = assemble(n, 45.0, method, "sinesine", principal)
            kxx, kxy, kyy = system.k
            if not np.allclose((kxx, kxy, kyy), (5.5, 4.5, 5.5),
                               rtol=0., atol=2e-14):
                raise AssertionError("incorrect physical conductivity tensor")
            # A genuine Dirichlet-eliminated system is solved; no fitted
            # manufactured-mode amplitude substitutes for this computation.
            u, linear_residual = nodal_solution(system)
            if (not math.isfinite(linear_residual)
                    or linear_residual > tolerances["linear_residual_linf_relative_max"]):
                raise AssertionError(f"{case_id} failed frozen linear residual gate: {linear_residual}")
            metrics = report_metrics(u, 45.0, method, "sinesine", principal)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError(f"nonfinite physical/scientific diagnostics in {case_id}")
            if (metrics["dual_cell_balance_scaled"] > tolerances["dual_cell_balance_scaled_max"]
                    or metrics["dual_boundary_budget_scaled"] > tolerances["dual_cell_balance_scaled_max"]):
                raise AssertionError(f"{case_id} violates the frozen Omega_h budget")
            # Verify both methods received the SAME complete tensor physical
            # right-hand side, including -9*pi^2*cos(pi*x)*cos(pi*y).
            pair_sources[method] = system.b.copy()
            if method == "M1" and not np.array_equal(pair_sources["M0"], system.b):
                raise AssertionError("M0 and M1 did not receive identical FULL-tensor source")
            raw = output / "raw" / case_id
            raw.mkdir(parents=True, exist_ok=False)
            if n in stage["save_full_fields_N"]:
                qx, qy = dual_face_flux(u, 45.0, method, principal)
                save_fields(raw, u, qx, qy)
            write_json(raw / "run.json", {
                "article_id": "tutorial-007", "experiment": STAGE,
                "case_id": case_id, "status": "solved",
                "field": "sinesine", "theta_deg": 45.0, "N": n,
                "method": method, "same_full_continuous_source_for_both_methods": True,
                "matrix_scaling": "unscaled_Lh_Dirichlet_eliminated",
                "control_volume_scope": "Omega_h_not_physical_boundary",
                "config_sha256": digest(config_path),
                "source_sha256": source_files,
                "source_commit": os.environ.get("GITHUB_SHA"),
                "scientific_csv_sha256": {
                    p.name: digest(p) for p in sorted(raw.glob("*.csv"))
                },
            })
            row = {
                "case_id": case_id, "experiment": STAGE, "field": "sinesine",
                "theta_deg": 45., "N": n, "h": 1.0/n, "method": method,
                "K_xx": kxx, "K_xy": kxy, "K_yy": kyy,
                "n_unknowns": (n-1)**2, "nnz": system.A.nnz,
                "linear_residual_linf_relative": linear_residual,
                **metrics, "solver_status": "solved",
            }
            recorded.append(row)
            print("ET007_03_SOLVED_REAL_SPARSE", case_id,
                  "T_L2", format(metrics["T_error_l2"], ".17g"),
                  "T_Linf", format(metrics["T_error_linf"], ".17g"),
                  "qx_Linf", format(metrics["q_x_error_linf"], ".17g"),
                  "qy_Linf", format(metrics["q_y_error_linf"], ".17g"),
                  "relative_residual", format(linear_residual, ".17g"), flush=True)

    if (len(recorded) != 8 or len(set(r["case_id"] for r in recorded)) != 8
            or [r["N"] for r in recorded] != [16,16,32,32,64,64,128,128]):
        raise AssertionError("missing, duplicated, or reordered ET007-03 case")
    write_csv(output / "summary.csv", SUMMARY, recorded)
    scientific_files = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output / "run.json", {
        "article_id": "tutorial-007", "experiment": STAGE,
        "status": "completed", "executed_case_count": 8,
        "planned_full_case_count": 28, "complete_28_case_program": False,
        "run_case_ids": [r["case_id"] for r in recorded],
        "full_tensor_source_shared": True,
        "config_sha256": digest(config_path), "source_sha256": source_files,
        "source_commit": os.environ.get("GITHUB_SHA"),
        "scientific_sha256": {
            str(p.relative_to(output)): digest(p) for p in scientific_files
        },
    })
    print("ET007_03_PRODUCTION_COMPLETED", len(recorded), "cases",
          len(list(output.rglob("*.csv"))), "scientific CSV", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=[STAGE])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
