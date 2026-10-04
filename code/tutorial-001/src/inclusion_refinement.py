#!/usr/bin/env python3
"""ET001-04: central rectangular inclusion with four-level refinement.

The verified harmonic TPFA core in composite_heat.py is frozen and imported
unchanged. This driver only constructs the inclusion conductivity field,
invokes the frozen assembly/boundary routines, and records ET001-04 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

import composite_heat as core


FROZEN_CORE_BLOB = "be4d7e82ce473c0259017813a2b2b5f580d891fa"


def aligned_face_index(value: float, spacing: float, name: str) -> int:
    ratio = value / spacing
    nearest = round(ratio)
    if abs(ratio - nearest) > 5e-12:
        raise ValueError(
            f"{name} must align with a grid face; "
            f"{name}/spacing={ratio:.17g}"
        )
    return int(nearest)


def inclusion_conductivity(
    config: dict[str, Any],
    grid: core.Grid,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    k_matrix = core.finite_positive(config["k_matrix"], "k_matrix")
    k_inclusion = core.finite_positive(
        config["k_inclusion"], "k_inclusion"
    )
    x0 = core.finite_value(config["inclusion"]["x0"], "inclusion.x0")
    x1 = core.finite_value(config["inclusion"]["x1"], "inclusion.x1")
    y0 = core.finite_value(config["inclusion"]["y0"], "inclusion.y0")
    y1 = core.finite_value(config["inclusion"]["y1"], "inclusion.y1")

    if not (0.0 < x0 < x1 < grid.lx):
        raise ValueError("inclusion x-bounds must lie strictly inside domain")
    if not (0.0 < y0 < y1 < grid.ly):
        raise ValueError("inclusion y-bounds must lie strictly inside domain")

    i0 = aligned_face_index(x0, grid.dx, "inclusion.x0")
    i1 = aligned_face_index(x1, grid.dx, "inclusion.x1")
    j0 = aligned_face_index(y0, grid.dy, "inclusion.y0")
    j1 = aligned_face_index(y1, grid.dy, "inclusion.y1")
    if not (0 < i0 < i1 < grid.nx and 0 < j0 < j1 < grid.ny):
        raise ValueError("inclusion face indices must be strictly internal")

    conductivity = np.full(
        (grid.ny, grid.nx), k_matrix, dtype=float
    )
    conductivity[j0:j1, i0:i1] = k_inclusion
    return conductivity, (i0, i1, j0, j1)


def solve_inclusion(
    config: dict[str, Any],
    nx: int,
    ny: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grid = core.Grid(
        nx=int(nx),
        ny=int(ny),
        lx=core.finite_positive(config["lx"], "lx"),
        ly=core.finite_positive(config["ly"], "ly"),
        thickness=core.finite_positive(
            config["thickness"], "thickness"
        ),
    )
    conductivity, bounds = inclusion_conductivity(config, grid)
    i0, i1, j0, j1 = bounds

    # Reuse the frozen TPFA harmonic assembly and boundary treatment.
    matrix, rhs = core.assemble_system(config, grid, conductivity)
    solution = spsolve(matrix, rhs)
    if (
        solution.shape != (grid.nx * grid.ny,)
        or not np.all(np.isfinite(solution))
    ):
        raise RuntimeError("ET001-04 sparse solve returned invalid values")

    residual = matrix @ solution - rhs
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    rhs_scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
    residual_relative_inf = residual_inf / rhs_scale
    temperature = solution.reshape((grid.ny, grid.nx))

    faces = core.reconstruct_faces(
        config, grid, temperature, conductivity
    )
    q_hot = sum(
        float(face["flux_positive_axis"])
        for face in faces
        if face["face_type"] == "boundary_left"
    )
    q_cold = sum(
        float(face["flux_positive_axis"])
        for face in faces
        if face["face_type"] == "boundary_right"
    )
    balance_relative = abs(q_hot - q_cold) / max(
        abs(q_hot), abs(q_cold), 1e-300
    )

    inclusion_temperature = temperature[j0:j1, i0:i1]
    inclusion_mean_temperature = float(
        np.mean(inclusion_temperature)
    )
    temperature_min = float(np.min(temperature))
    temperature_max = float(np.max(temperature))

    column_ranges = np.max(temperature, axis=0) - np.min(
        temperature, axis=0
    )
    max_column_temperature_range = float(np.max(column_ranges))

    internal_y_fluxes = [
        abs(float(face["flux_positive_axis"]))
        for face in faces
        if face["face_type"] == "internal_y"
    ]
    max_abs_internal_y_flux = max(internal_y_fluxes, default=0.0)

    symmetry_target = float(config["t_hot"]) + float(config["t_cold"])
    symmetry_error = float(
        np.max(
            np.abs(
                temperature
                + np.fliplr(temperature)
                - symmetry_target
            )
        )
    )

    interface_face_count = sum(
        1
        for face in faces
        if face["face_type"] in {"internal_x", "internal_y"}
        and float(face["k_p"]) != float(face["k_n"])
    )

    summary = {
        "experiment_id": "ET001-04",
        "case_name": config["case_name"],
        "nx": grid.nx,
        "ny": grid.ny,
        "dx": grid.dx,
        "dy": grid.dy,
        "unknowns": grid.nx * grid.ny,
        "matrix_nnz": int(matrix.nnz),
        "inclusion_i0": i0,
        "inclusion_i1": i1,
        "inclusion_j0": j0,
        "inclusion_j1": j1,
        "inclusion_cell_count": (i1 - i0) * (j1 - j0),
        "interface_face_count": interface_face_count,
        "temperature_min": temperature_min,
        "temperature_max": temperature_max,
        "inclusion_mean_temperature": inclusion_mean_temperature,
        "q_hot": q_hot,
        "q_cold": q_cold,
        "global_balance_relative": balance_relative,
        "residual_inf": residual_inf,
        "residual_relative_inf": residual_relative_inf,
        "symmetry_error": symmetry_error,
        "max_column_temperature_range": max_column_temperature_range,
        "max_abs_internal_y_flux": max_abs_internal_y_flux,
    }

    # Frozen ET001-04 gates that can be evaluated per grid.
    bound_tol = 1e-9
    if temperature_min < float(config["t_cold"]) - bound_tol:
        raise AssertionError("ET001-04 temperature fell below cold boundary")
    if temperature_max > float(config["t_hot"]) + bound_tol:
        raise AssertionError("ET001-04 temperature exceeded hot boundary")
    if balance_relative > 1e-10:
        raise AssertionError("ET001-04 global heat balance exceeds 1e-10")
    if residual_relative_inf > 1e-11:
        raise AssertionError("ET001-04 relative residual exceeds 1e-11")

    cell_rows: list[dict[str, Any]] = []
    for j in range(grid.ny):
        for i in range(grid.nx):
            in_inclusion = i0 <= i < i1 and j0 <= j < j1
            cell_rows.append(
                {
                    "experiment_id": "ET001-04",
                    "case_name": config["case_name"],
                    "nx": grid.nx,
                    "ny": grid.ny,
                    "i": i,
                    "j": j,
                    "x": (i + 0.5) * grid.dx,
                    "y": (j + 0.5) * grid.dy,
                    "material": (
                        "inclusion" if in_inclusion else "matrix"
                    ),
                    "k": float(conductivity[j, i]),
                    "temperature": float(temperature[j, i]),
                }
            )

    face_rows: list[dict[str, Any]] = []
    for face in faces:
        k_n = face["k_n"]
        is_interface = (
            face["face_type"] in {"internal_x", "internal_y"}
            and k_n != ""
            and float(face["k_p"]) != float(k_n)
        )
        face_rows.append(
            {
                "experiment_id": "ET001-04",
                "case_name": config["case_name"],
                "nx": grid.nx,
                "ny": grid.ny,
                **face,
                "is_material_interface": int(is_interface),
            }
        )

    return summary, cell_rows, face_rows


def relative_difference(fine: float, coarse: float) -> float:
    return abs(fine - coarse) / max(abs(fine), 1e-300)


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


def run(config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "ET001-04":
        raise ValueError("inclusion_refinement.py only accepts ET001-04")

    summaries: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_faces: list[dict[str, Any]] = []

    for pair in config["grids"]:
        nx, ny = int(pair[0]), int(pair[1])
        summary, cells, faces = solve_inclusion(config, nx, ny)
        summaries.append(summary)
        all_cells.extend(cells)
        all_faces.extend(faces)

    refinement_rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for summary in summaries:
        if previous is None:
            q_change = ""
            mean_change = ""
        else:
            q_change = relative_difference(
                float(summary["q_hot"]), float(previous["q_hot"])
            )
            mean_change = relative_difference(
                float(summary["inclusion_mean_temperature"]),
                float(previous["inclusion_mean_temperature"]),
            )
        refinement_rows.append(
            {
                "experiment_id": "ET001-04",
                "nx": summary["nx"],
                "ny": summary["ny"],
                "q_hot": summary["q_hot"],
                "inclusion_mean_temperature": (
                    summary["inclusion_mean_temperature"]
                ),
                "q_hot_relative_change_from_previous": q_change,
                "inclusion_mean_relative_change_from_previous": mean_change,
            }
        )
        previous = summary

    if len(summaries) < 2:
        raise ValueError("ET001-04 requires at least two refinement levels")
    fine = summaries[-1]
    next_coarse = summaries[-2]
    q_last_difference = relative_difference(
        float(fine["q_hot"]), float(next_coarse["q_hot"])
    )
    mean_last_difference = relative_difference(
        float(fine["inclusion_mean_temperature"]),
        float(next_coarse["inclusion_mean_temperature"]),
    )
    if q_last_difference > 0.005:
        raise AssertionError(
            "ET001-04 last-two q_hot relative difference exceeds 0.5%"
        )
    if mean_last_difference > 0.005:
        raise AssertionError(
            "ET001-04 last-two inclusion mean relative difference "
            "exceeds 0.5%"
        )

    summary_fields = list(summaries[0].keys())
    write_csv(output_dir / "summary.csv", summaries, summary_fields)

    cell_fields = list(all_cells[0].keys())
    write_csv(output_dir / "cells.csv", all_cells, cell_fields)

    face_fields = list(all_faces[0].keys())
    write_csv(output_dir / "faces.csv", all_faces, face_fields)

    refinement_fields = list(refinement_rows[0].keys())
    write_csv(
        output_dir / "refinement.csv",
        refinement_rows,
        refinement_fields,
    )

    run_lines = [
        "experiment_id=ET001-04",
        f"config={config_path.as_posix()}",
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"platform={platform.platform()}",
        f"frozen_core_blob={FROZEN_CORE_BLOB}",
        "harmonic_path=imported unchanged from composite_heat.py",
        "solver=scipy.sparse.linalg.spsolve",
        "status=PASS",
        f"last_two_q_hot_relative_difference={q_last_difference:.17g}",
        (
            "last_two_inclusion_mean_relative_difference="
            f"{mean_last_difference:.17g}"
        ),
    ]
    for summary in summaries:
        run_lines.append(
            f"grid={summary['nx']}x{summary['ny']} "
            f"unknowns={summary['unknowns']} "
            f"nnz={summary['matrix_nnz']} "
            f"q_hot={float(summary['q_hot']):.17g} "
            f"q_cold={float(summary['q_cold']):.17g} "
            f"inclusion_mean="
            f"{float(summary['inclusion_mean_temperature']):.17g} "
            f"temperature_min={float(summary['temperature_min']):.17g} "
            f"temperature_max={float(summary['temperature_max']):.17g} "
            f"balance_relative="
            f"{float(summary['global_balance_relative']):.17g} "
            f"residual_relative_inf="
            f"{float(summary['residual_relative_inf']):.17g} "
            f"symmetry_error={float(summary['symmetry_error']):.17g} "
            f"max_column_temperature_range="
            f"{float(summary['max_column_temperature_range']):.17g} "
            f"max_abs_internal_y_flux="
            f"{float(summary['max_abs_internal_y_flux']):.17g}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.txt").write_text(
        "\n".join(run_lines) + "\n", encoding="utf-8"
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = run(args.config, args.output)
    for summary in summaries:
        print(
            f"ET001-04 {summary['nx']}x{summary['ny']} "
            f"q_hot={float(summary['q_hot']):.12g} "
            f"inclusion_mean="
            f"{float(summary['inclusion_mean_temperature']):.12g} "
            f"balance_rel="
            f"{float(summary['global_balance_relative']):.3e} "
            f"residual_rel="
            f"{float(summary['residual_relative_inf']):.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
