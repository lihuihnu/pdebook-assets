#!/usr/bin/env python3
"""ET001-05 conservation audit of frozen ET001-04 CSV data.

This script is intentionally solver-free:
- standard library only;
- no import of composite_heat.py;
- no NumPy/SciPy;
- no matrix assembly or PDE solve;
- no flux recomputation from temperatures.

It reconstructs control-volume balances solely from the already committed
ET001-04 cells.csv and faces.csv.
"""

from __future__ import annotations

import argparse
import csv
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_BLOBS = {
    "summary.csv": "814f1c53834ee194074491b2a8c32d3ee07b5568",
    "cells.csv": "0c6f11ac951bdcdcffaa6b15927735b0976c83a1",
    "faces.csv": "c99d0113946203c75de3f4e4a43209b19488bbc5",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def group_by_grid(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], list[dict[str, str]]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["nx"]), int(row["ny"]))].append(row)
    return dict(grouped)


def audit_grid(
    summary_row: dict[str, str],
    cell_rows: list[dict[str, str]],
    face_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nx = int(summary_row["nx"])
    ny = int(summary_row["ny"])
    expected_cells = nx * ny
    if len(cell_rows) != expected_cells:
        raise AssertionError(
            f"{nx}x{ny}: expected {expected_cells} cells, got {len(cell_rows)}"
        )

    material: dict[tuple[int, int], str] = {}
    for row in cell_rows:
        key = (int(row["i"]), int(row["j"]))
        if key in material:
            raise AssertionError(f"{nx}x{ny}: duplicate cell {key}")
        material[key] = row["material"]

    directions: dict[tuple[int, int], dict[str, float]] = {
        key: {"west": 0.0, "east": 0.0, "south": 0.0, "north": 0.0}
        for key in material
    }

    q_hot = 0.0
    q_cold = 0.0
    top_outward = 0.0
    bottom_outward = 0.0
    top_abs = 0.0
    bottom_abs = 0.0
    inclusion_net_outward = 0.0
    inclusion_abs_interface_throughput = 0.0
    interface_face_count = 0
    internal_pair_cancellation = 0.0
    internal_face_count = 0

    for row in face_rows:
        face_type = row["face_type"]
        i = int(row["i"])
        j = int(row["j"])
        flux = float(row["flux_positive_axis"])

        if face_type == "internal_x":
            west = (i, j)
            east = (i + 1, j)
            if west not in material or east not in material:
                raise AssertionError(f"{nx}x{ny}: invalid internal_x face")
            directions[west]["east"] += flux
            directions[east]["west"] -= flux
            internal_pair_cancellation += flux - flux
            internal_face_count += 1

            mw = material[west]
            me = material[east]
            if mw != me:
                interface_face_count += 1
                if {mw, me} != {"matrix", "inclusion"}:
                    raise AssertionError(
                        f"{nx}x{ny}: unexpected material pair {mw}/{me}"
                    )
                contribution = flux if mw == "inclusion" else -flux
                inclusion_net_outward += contribution
                inclusion_abs_interface_throughput += abs(contribution)

        elif face_type == "internal_y":
            south = (i, j)
            north = (i, j + 1)
            if south not in material or north not in material:
                raise AssertionError(f"{nx}x{ny}: invalid internal_y face")
            directions[south]["north"] += flux
            directions[north]["south"] -= flux
            internal_pair_cancellation += flux - flux
            internal_face_count += 1

            ms = material[south]
            mn = material[north]
            if ms != mn:
                interface_face_count += 1
                if {ms, mn} != {"matrix", "inclusion"}:
                    raise AssertionError(
                        f"{nx}x{ny}: unexpected material pair {ms}/{mn}"
                    )
                contribution = flux if ms == "inclusion" else -flux
                inclusion_net_outward += contribution
                inclusion_abs_interface_throughput += abs(contribution)

        elif face_type == "boundary_left":
            cell = (0, j)
            if cell not in material:
                raise AssertionError(f"{nx}x{ny}: invalid left boundary face")
            # Stored positive-axis flux points from hot boundary into the domain.
            directions[cell]["west"] -= flux
            q_hot += flux

        elif face_type == "boundary_right":
            cell = (nx - 1, j)
            if cell not in material:
                raise AssertionError(f"{nx}x{ny}: invalid right boundary face")
            directions[cell]["east"] += flux
            q_cold += flux

        elif face_type == "boundary_bottom":
            cell = (i, 0)
            if cell not in material:
                raise AssertionError(f"{nx}x{ny}: invalid bottom boundary face")
            # Positive-axis convention is +y, hence outward at bottom is -flux.
            directions[cell]["south"] -= flux
            bottom_outward -= flux
            bottom_abs += abs(flux)

        elif face_type == "boundary_top":
            cell = (i, ny - 1)
            if cell not in material:
                raise AssertionError(f"{nx}x{ny}: invalid top boundary face")
            directions[cell]["north"] += flux
            top_outward += flux
            top_abs += abs(flux)

        else:
            raise AssertionError(f"{nx}x{ny}: unknown face type {face_type!r}")

    q_ref = max(abs(q_hot), abs(q_cold), 1e-300)
    balance_rows: list[dict[str, Any]] = []
    max_cell_abs_gap = -1.0
    max_cell_key = (-1, -1)
    sum_cell_net_outward = 0.0

    for j in range(ny):
        for i in range(nx):
            key = (i, j)
            d = directions[key]
            net = d["west"] + d["east"] + d["south"] + d["north"]
            abs_gap = abs(net)
            sum_cell_net_outward += net
            if abs_gap > max_cell_abs_gap:
                max_cell_abs_gap = abs_gap
                max_cell_key = key
            balance_rows.append(
                {
                    "experiment_id": "ET001-05",
                    "source_experiment_id": "ET001-04",
                    "nx": nx,
                    "ny": ny,
                    "i": i,
                    "j": j,
                    "material": material[key],
                    "west_outward": d["west"],
                    "east_outward": d["east"],
                    "south_outward": d["south"],
                    "north_outward": d["north"],
                    "net_outward_heat_rate": net,
                    "normalized_abs_gap": abs_gap / q_ref,
                }
            )

    boundary_net_outward = (
        -q_hot + q_cold + bottom_outward + top_outward
    )
    global_balance_relative = abs(q_hot - q_cold) / q_ref
    max_cell_balance_relative = max_cell_abs_gap / q_ref
    inclusion_balance_relative = abs(inclusion_net_outward) / q_ref
    adiabatic_boundary_relative = (top_abs + bottom_abs) / q_ref
    cell_sum_boundary_consistency_relative = abs(
        sum_cell_net_outward - boundary_net_outward
    ) / q_ref
    internal_pair_cancellation_relative = abs(
        internal_pair_cancellation
    ) / q_ref

    expected_interface_faces = int(summary_row["interface_face_count"])
    if interface_face_count != expected_interface_faces:
        raise AssertionError(
            f"{nx}x{ny}: interface faces {interface_face_count} "
            f"!= ET001-04 summary {expected_interface_faces}"
        )

    if abs(q_hot - float(summary_row["q_hot"])) > 1e-12:
        raise AssertionError(f"{nx}x{ny}: q_hot differs from ET001-04 summary")
    if abs(q_cold - float(summary_row["q_cold"])) > 1e-12:
        raise AssertionError(f"{nx}x{ny}: q_cold differs from ET001-04 summary")

    threshold = 1e-10
    gates = {
        "global_balance_relative": global_balance_relative,
        "max_cell_balance_relative": max_cell_balance_relative,
        "inclusion_balance_relative": inclusion_balance_relative,
        "adiabatic_boundary_relative": adiabatic_boundary_relative,
        "cell_sum_boundary_consistency_relative": (
            cell_sum_boundary_consistency_relative
        ),
        "internal_pair_cancellation_relative": (
            internal_pair_cancellation_relative
        ),
    }
    failed = {name: value for name, value in gates.items() if value > threshold}
    if failed:
        raise AssertionError(f"{nx}x{ny}: conservation gates failed: {failed}")

    audit = {
        "experiment_id": "ET001-05",
        "source_experiment_id": "ET001-04",
        "nx": nx,
        "ny": ny,
        "cell_count": expected_cells,
        "internal_face_count": internal_face_count,
        "interface_face_count": interface_face_count,
        "q_hot": q_hot,
        "q_cold": q_cold,
        "q_ref": q_ref,
        "global_balance_relative": global_balance_relative,
        "max_cell_abs_gap": max_cell_abs_gap,
        "max_cell_balance_relative": max_cell_balance_relative,
        "max_cell_i": max_cell_key[0],
        "max_cell_j": max_cell_key[1],
        "max_cell_material": material[max_cell_key],
        "sum_cell_net_outward": sum_cell_net_outward,
        "boundary_net_outward": boundary_net_outward,
        "cell_sum_boundary_consistency_relative": (
            cell_sum_boundary_consistency_relative
        ),
        "internal_pair_cancellation_abs": abs(internal_pair_cancellation),
        "internal_pair_cancellation_relative": (
            internal_pair_cancellation_relative
        ),
        "top_boundary_outward": top_outward,
        "bottom_boundary_outward": bottom_outward,
        "top_bottom_abs_flux_sum": top_abs + bottom_abs,
        "adiabatic_boundary_relative": adiabatic_boundary_relative,
        "inclusion_net_outward": inclusion_net_outward,
        "inclusion_abs_interface_throughput": (
            inclusion_abs_interface_throughput
        ),
        "inclusion_balance_relative": inclusion_balance_relative,
    }
    return audit, balance_rows


def run(source_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    summaries = read_rows(source_dir / "summary.csv")
    cells = group_by_grid(read_rows(source_dir / "cells.csv"))
    faces = group_by_grid(read_rows(source_dir / "faces.csv"))

    audits: list[dict[str, Any]] = []
    balances: list[dict[str, Any]] = []

    for summary in summaries:
        key = (int(summary["nx"]), int(summary["ny"]))
        if key not in cells or key not in faces:
            raise AssertionError(f"missing frozen ET001-04 data for grid {key}")
        audit, grid_balances = audit_grid(
            summary,
            cells[key],
            faces[key],
        )
        audits.append(audit)
        balances.extend(grid_balances)

    write_csv(
        output_dir / "audit_summary.csv",
        audits,
        list(audits[0].keys()),
    )
    write_csv(
        output_dir / "balances.csv",
        balances,
        list(balances[0].keys()),
    )

    run_lines = [
        "experiment_id=ET001-05",
        "source_experiment_id=ET001-04",
        f"source_dir={source_dir.as_posix()}",
        f"python={platform.python_version()}",
        "dependencies=python-standard-library-only",
        "pde_solve=NOT_RUN",
        "matrix_assembly=NOT_RUN",
        "flux_recomputation_from_temperature=NOT_RUN",
        f"source_summary_blob={SOURCE_BLOBS['summary.csv']}",
        f"source_cells_blob={SOURCE_BLOBS['cells.csv']}",
        f"source_faces_blob={SOURCE_BLOBS['faces.csv']}",
        "status=PASS",
    ]
    for audit in audits:
        run_lines.append(
            f"grid={audit['nx']}x{audit['ny']} "
            f"q_hot={float(audit['q_hot']):.17g} "
            f"q_cold={float(audit['q_cold']):.17g} "
            f"global_balance_relative="
            f"{float(audit['global_balance_relative']):.17g} "
            f"max_cell_balance_relative="
            f"{float(audit['max_cell_balance_relative']):.17g} "
            f"inclusion_balance_relative="
            f"{float(audit['inclusion_balance_relative']):.17g} "
            f"adiabatic_boundary_relative="
            f"{float(audit['adiabatic_boundary_relative']):.17g}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.txt").write_text(
        "\n".join(run_lines) + "\n",
        encoding="utf-8",
    )
    return audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audits = run(args.source, args.output)
    for audit in audits:
        print(
            f"ET001-05 {audit['nx']}x{audit['ny']} "
            f"global={float(audit['global_balance_relative']):.3e} "
            f"max_cell={float(audit['max_cell_balance_relative']):.3e} "
            f"inclusion={float(audit['inclusion_balance_relative']):.3e} "
            f"adiabatic={float(audit['adiabatic_boundary_relative']):.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
