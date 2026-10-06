#!/usr/bin/env python3
"""Run ET002-01: P1 assembly and linear patch test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fem_core import build_l_shape_mesh, mesh_quality, solve_dirichlet_laplace


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/et002-01.json"
RESULT_DIR = ROOT / "results/et002-01"


def patch_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 + x - 2.0 * y


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in config["meshes"]:
        mesh = build_l_shape_mesh(int(spec["N"]), float(spec["beta"]))
        result = solve_dirichlet_laplace(mesh, patch_value)
        exact = patch_value(mesh.points[:, 0], mesh.points[:, 1])
        max_error = float(np.max(np.abs(result.values - exact)))
        quality = mesh_quality(mesh)

        row = {
            "mesh": spec["name"],
            "N": mesh.N,
            "beta": mesh.beta,
            "nodes": mesh.points.shape[0],
            "triangles": mesh.triangles.shape[0],
            "boundary_nodes": mesh.boundary_nodes.shape[0],
            "free_nodes": result.free_nodes.shape[0],
            "min_signed_area": result.assembly_audit.min_signed_area,
            "max_local_symmetry_defect": result.assembly_audit.max_local_symmetry_defect,
            "max_local_row_sum_defect": result.assembly_audit.max_local_row_sum_defect,
            "free_block_symmetry_defect": result.free_block_symmetry_defect,
            "max_nodal_error": max_error,
            "reduced_residual_l2": result.reduced_residual_l2,
            "min_angle_deg": quality["min_angle_deg"],
            "max_edge_ratio": quality["max_edge_ratio"],
        }
        rows.append(row)

        if not row["min_signed_area"] > 0.0:
            raise AssertionError(f"{spec['name']}: non-positive triangle")
        if max_error > float(config["max_nodal_error"]):
            raise AssertionError(f"{spec['name']}: patch error {max_error}")
        if result.reduced_residual_l2 > float(config["max_reduced_residual_l2"]):
            raise AssertionError(f"{spec['name']}: residual {result.reduced_residual_l2}")
        if result.assembly_audit.max_local_symmetry_defect > float(config["max_local_symmetry_defect"]):
            raise AssertionError(f"{spec['name']}: local symmetry defect")
        if result.assembly_audit.max_local_row_sum_defect > float(config["max_local_row_sum_defect"]):
            raise AssertionError(f"{spec['name']}: local row-sum defect")
        if result.free_block_symmetry_defect > float(config["max_free_block_symmetry_defect"]):
            raise AssertionError(f"{spec['name']}: free block symmetry defect")

    fieldnames = list(rows[0].keys())
    with (RESULT_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("ET002-01 PASS")
    for row in rows:
        print(
            f"{row['mesh']}: nodes={row['nodes']} triangles={row['triangles']} "
            f"max_nodal_error={row['max_nodal_error']:.16e} "
            f"residual={row['reduced_residual_l2']:.16e}"
        )


if __name__ == "__main__":
    main()
