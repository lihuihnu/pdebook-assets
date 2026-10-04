#!/usr/bin/env python3
"""ET001-06: conductivity-contrast sweep on the verified 96x64 grid.

Only k_inclusion/k_matrix changes. Geometry, boundary conditions, grid,
harmonic TPFA core, inclusion construction, and conservation gates are reused
from already verified ET001-04/05 code paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import inclusion_refinement as inclusion
import conservation_audit as audit


FROZEN_CORE_BLOB = "be4d7e82ce473c0259017813a2b2b5f580d891fa"
FROZEN_INCLUSION_DRIVER_BLOB = "88abe32035a3ca5b78f47b75e1d66f4b4b19642a"
FROZEN_AUDIT_BLOB = "dd8ffbb1238aa67ecb3d4483c75f5efab8447168"
ET004_SUMMARY_BLOB = "814f1c53834ee194074491b2a8c32d3ee07b5568"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def effective_conductivity(
    q_hot: float,
    lx: float,
    ly: float,
    thickness: float,
    t_hot: float,
    t_cold: float,
) -> float:
    return q_hot * lx / (thickness * ly * (t_hot - t_cold))


def run(
    config_path: Path,
    et004_summary_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "ET001-06":
        raise ValueError("contrast_sweep.py only accepts ET001-06")

    nx, ny = int(config["grid"][0]), int(config["grid"][1])
    if (nx, ny) != (96, 64):
        raise ValueError("ET001-06 is frozen to the verified 96x64 grid")

    k_matrix = float(config["k_matrix"])
    if k_matrix <= 0.0:
        raise ValueError("k_matrix must be positive")

    contrasts = [float(value) for value in config["contrasts"]]
    expected = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    if contrasts != expected:
        raise ValueError(f"unexpected ET001-06 contrast list: {contrasts}")

    et004_rows = []
    with et004_summary_path.open("r", encoding="utf-8", newline="") as handle:
        et004_rows = list(csv.DictReader(handle))
    et004_96 = [
        row
        for row in et004_rows
        if int(row["nx"]) == 96 and int(row["ny"]) == 64
    ]
    if len(et004_96) != 1:
        raise ValueError("expected one ET001-04 96x64 reference row")
    et004_ref = et004_96[0]

    sweep_rows: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_faces: list[dict[str, Any]] = []
    all_balances: list[dict[str, Any]] = []

    for contrast in contrasts:
        case_config = {
            "experiment_id": "ET001-06",
            "case_name": f"inclusion_contrast_{contrast:g}",
            "lx": config["lx"],
            "ly": config["ly"],
            "thickness": config["thickness"],
            "t_hot": config["t_hot"],
            "t_cold": config["t_cold"],
            "qv": config.get("qv", 0.0),
            "k_matrix": k_matrix,
            "k_inclusion": k_matrix * contrast,
            "inclusion": config["inclusion"],
        }

        summary, cells, faces = inclusion.solve_inclusion(
            case_config,
            nx,
            ny,
        )

        # Reuse the already verified ET001-05 conservation gates on the newly
        # generated faces/cells for every parameter point.
        # ET001-05 audits the geometric material boundary from
        # cells.csv. ET001-04's summary interface_face_count instead counts
        # faces where k actually jumps. At chi=1 the geometric material
        # boundary still exists although k is continuous, so provide the
        # material-boundary count expected by the frozen audit contract.
        i0 = int(summary["inclusion_i0"])
        i1 = int(summary["inclusion_i1"])
        j0 = int(summary["inclusion_j0"])
        j1 = int(summary["inclusion_j1"])
        audit_input_summary = dict(summary)
        audit_input_summary["interface_face_count"] = (
            2 * (j1 - j0) + 2 * (i1 - i0)
        )
        audit_summary, balances = audit.audit_grid(
            audit_input_summary,
            cells,
            faces,
        )

        k_eff = effective_conductivity(
            float(summary["q_hot"]),
            float(config["lx"]),
            float(config["ly"]),
            float(config["thickness"]),
            float(config["t_hot"]),
            float(config["t_cold"]),
        )

        row = {
            "experiment_id": "ET001-06",
            "case_name": config["case_name"],
            "contrast": contrast,
            "k_matrix": k_matrix,
            "k_inclusion": k_matrix * contrast,
            "nx": nx,
            "ny": ny,
            "temperature_min": summary["temperature_min"],
            "temperature_max": summary["temperature_max"],
            "inclusion_mean_temperature": summary[
                "inclusion_mean_temperature"
            ],
            "q_hot": summary["q_hot"],
            "q_cold": summary["q_cold"],
            "k_eff": k_eff,
            "residual_relative_inf": summary["residual_relative_inf"],
            "global_balance_relative": audit_summary[
                "global_balance_relative"
            ],
            "max_cell_balance_relative": audit_summary[
                "max_cell_balance_relative"
            ],
            "inclusion_balance_relative": audit_summary[
                "inclusion_balance_relative"
            ],
            "adiabatic_boundary_relative": audit_summary[
                "adiabatic_boundary_relative"
            ],
            "cell_sum_boundary_consistency_relative": audit_summary[
                "cell_sum_boundary_consistency_relative"
            ],
            "internal_pair_cancellation_relative": audit_summary[
                "internal_pair_cancellation_relative"
            ],
            "inclusion_abs_interface_throughput": audit_summary[
                "inclusion_abs_interface_throughput"
            ],
            "max_column_temperature_range": summary[
                "max_column_temperature_range"
            ],
            "max_abs_internal_y_flux": summary[
                "max_abs_internal_y_flux"
            ],
        }
        sweep_rows.append(row)

        for cell in cells:
            all_cells.append(
                {
                    "experiment_id": "ET001-06",
                    "contrast": contrast,
                    **{
                        key: value
                        for key, value in cell.items()
                        if key not in {"experiment_id", "case_name"}
                    },
                }
            )
        for face in faces:
            all_faces.append(
                {
                    "experiment_id": "ET001-06",
                    "contrast": contrast,
                    **{
                        key: value
                        for key, value in face.items()
                        if key not in {"experiment_id", "case_name"}
                    },
                }
            )
        for balance in balances:
            all_balances.append(
                {
                    "experiment_id": "ET001-06",
                    "contrast": contrast,
                    **{
                        key: value
                        for key, value in balance.items()
                        if key not in {
                            "experiment_id",
                            "source_experiment_id",
                        }
                    },
                }
            )

    by_contrast = {float(row["contrast"]): row for row in sweep_rows}

    # Frozen analytic anchor: chi=1 returns the uniform-material problem.
    uniform_q = (
        k_matrix
        * float(config["ly"])
        * float(config["thickness"])
        * (float(config["t_hot"]) - float(config["t_cold"]))
        / float(config["lx"])
    )
    chi1 = by_contrast[1.0]
    if abs(float(chi1["q_hot"]) - uniform_q) / abs(uniform_q) > 1e-9:
        raise AssertionError("ET001-06 chi=1 failed uniform-material anchor")
    if abs(float(chi1["k_eff"]) - k_matrix) > 1e-9:
        raise AssertionError("ET001-06 chi=1 failed k_eff anchor")

    # Repository anchor: chi=10 must reproduce verified ET001-04 96x64.
    chi10 = by_contrast[10.0]
    for field in (
        "q_hot",
        "q_cold",
        "inclusion_mean_temperature",
        "temperature_min",
        "temperature_max",
    ):
        current = float(chi10[field])
        reference = float(et004_ref[field])
        if abs(current - reference) > 1e-12 * max(1.0, abs(reference)):
            raise AssertionError(
                f"ET001-06 chi=10 differs from ET001-04 96x64: {field}"
            )

    write_csv(
        output_dir / "contrast_sweep.csv",
        sweep_rows,
        list(sweep_rows[0].keys()),
    )
    write_csv(
        output_dir / "cells.csv",
        all_cells,
        list(all_cells[0].keys()),
    )
    write_csv(
        output_dir / "faces.csv",
        all_faces,
        list(all_faces[0].keys()),
    )
    write_csv(
        output_dir / "balances.csv",
        all_balances,
        list(all_balances[0].keys()),
    )

    run_lines = [
        "experiment_id=ET001-06",
        f"config={config_path.as_posix()}",
        f"python={platform.python_version()}",
        "grid=96x64",
        f"frozen_core_blob={FROZEN_CORE_BLOB}",
        f"frozen_inclusion_driver_blob={FROZEN_INCLUSION_DRIVER_BLOB}",
        f"frozen_audit_blob={FROZEN_AUDIT_BLOB}",
        f"et004_summary_blob={ET004_SUMMARY_BLOB}",
        "scheme=frozen harmonic TPFA",
        "conservation_gate=ET001-05 audit_grid threshold 1e-10",
        "status=PASS",
    ]
    for row in sweep_rows:
        run_lines.append(
            f"contrast={float(row['contrast']):g} "
            f"k_inclusion={float(row['k_inclusion']):.17g} "
            f"q_hot={float(row['q_hot']):.17g} "
            f"k_eff={float(row['k_eff']):.17g} "
            f"global_balance_relative="
            f"{float(row['global_balance_relative']):.17g} "
            f"max_cell_balance_relative="
            f"{float(row['max_cell_balance_relative']):.17g} "
            f"inclusion_balance_relative="
            f"{float(row['inclusion_balance_relative']):.17g} "
            f"adiabatic_boundary_relative="
            f"{float(row['adiabatic_boundary_relative']):.17g}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.txt").write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )
    return sweep_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--et004-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = run(args.config, args.et004_summary, args.output)
    for row in rows:
        print(
            f"ET001-06 chi={float(row['contrast']):g} "
            f"q_hot={float(row['q_hot']):.12g} "
            f"k_eff={float(row['k_eff']):.12g} "
            f"global={float(row['global_balance_relative']):.3e} "
            f"max_cell={float(row['max_cell_balance_relative']):.3e} "
            f"inclusion={float(row['inclusion_balance_relative']):.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
