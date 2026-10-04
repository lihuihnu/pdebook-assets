#!/usr/bin/env python3
"""Cell-centered FVM/TPFA solver for tutorial-001 ET001-01 and ET001-02.

This module intentionally implements only:
- ET001-01: uniform-material analytic regression.
- ET001-02: two-layer analytic interface benchmark.

It does not implement the later averaging ablation, inclusion refinement,
conservation audit, or contrast sweep experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    lx: float
    ly: float
    thickness: float

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.thickness

    @property
    def x_face_area(self) -> float:
        return self.dy * self.thickness

    @property
    def y_face_area(self) -> float:
        return self.dx * self.thickness


@dataclass
class SolveResult:
    grid: Grid
    temperature: np.ndarray
    conductivity: np.ndarray
    residual_inf: float
    residual_rel_inf: float
    q_hot: float
    q_cold: float
    faces: list[dict[str, Any]]


def finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def finite_value(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def cell_index(i: int, j: int, nx: int) -> int:
    return j * nx + i


def conductivity_field(case: dict[str, Any], grid: Grid) -> np.ndarray:
    x = (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    k = np.empty((grid.ny, grid.nx), dtype=float)

    kind = str(case["kind"])
    if kind == "uniform":
        k_value = finite_positive(case["k"], "k")
        k.fill(k_value)
        return k

    if kind == "layered":
        interface_x = finite_value(case["interface_x"], "interface_x")
        if not (0.0 < interface_x < grid.lx):
            raise ValueError("interface_x must lie strictly inside the domain")
        ratio = interface_x / grid.dx
        nearest = round(ratio)
        if abs(ratio - nearest) > 5e-12:
            raise ValueError(
                "layered interface must align with a vertical grid face; "
                f"interface_x/dx={ratio:.17g}"
            )
        k_left = finite_positive(case["k_left"], "k_left")
        k_right = finite_positive(case["k_right"], "k_right")
        for i, xc in enumerate(x):
            k[:, i] = k_left if xc < interface_x else k_right
        return k

    raise ValueError(f"unsupported case kind: {kind}")


def internal_conductance(
    area: float,
    distance_p: float,
    k_p: float,
    distance_n: float,
    k_n: float,
) -> float:
    """Distance-weighted harmonic conductance across one internal face."""
    return area / (distance_p / k_p + distance_n / k_n)


def boundary_conductance(area: float, distance: float, k_p: float) -> float:
    return k_p * area / distance


def assemble_system(
    config: dict[str, Any],
    grid: Grid,
    conductivity: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    unknown_count = grid.nx * grid.ny
    matrix = lil_matrix((unknown_count, unknown_count), dtype=float)
    rhs = np.zeros(unknown_count, dtype=float)
    qv = finite_value(config.get("qv", 0.0), "qv")
    t_hot = finite_value(config["t_hot"], "t_hot")
    t_cold = finite_value(config["t_cold"], "t_cold")
    rhs[:] = qv * grid.cell_volume

    # Each internal face is assembled once.
    for j in range(grid.ny):
        for i in range(grid.nx - 1):
            p = cell_index(i, j, grid.nx)
            n = cell_index(i + 1, j, grid.nx)
            g = internal_conductance(
                grid.x_face_area,
                0.5 * grid.dx,
                conductivity[j, i],
                0.5 * grid.dx,
                conductivity[j, i + 1],
            )
            matrix[p, p] += g
            matrix[n, n] += g
            matrix[p, n] -= g
            matrix[n, p] -= g

    for j in range(grid.ny - 1):
        for i in range(grid.nx):
            p = cell_index(i, j, grid.nx)
            n = cell_index(i, j + 1, grid.nx)
            g = internal_conductance(
                grid.y_face_area,
                0.5 * grid.dy,
                conductivity[j, i],
                0.5 * grid.dy,
                conductivity[j + 1, i],
            )
            matrix[p, p] += g
            matrix[n, n] += g
            matrix[p, n] -= g
            matrix[n, p] -= g

    # Dirichlet faces use the half-cell center-to-boundary distance.
    for j in range(grid.ny):
        p_left = cell_index(0, j, grid.nx)
        g_left = boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, conductivity[j, 0]
        )
        matrix[p_left, p_left] += g_left
        rhs[p_left] += g_left * t_hot

        p_right = cell_index(grid.nx - 1, j, grid.nx)
        g_right = boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, conductivity[j, -1]
        )
        matrix[p_right, p_right] += g_right
        rhs[p_right] += g_right * t_cold

    # Top/bottom are homogeneous Neumann, so they add no flux term.
    return matrix.tocsr(), rhs


def reconstruct_faces(
    config: dict[str, Any],
    grid: Grid,
    temperature: np.ndarray,
    conductivity: np.ndarray,
) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    t_hot = float(config["t_hot"])
    t_cold = float(config["t_cold"])

    # Internal x-faces; positive flux means west -> east.
    for j in range(grid.ny):
        for i in range(grid.nx - 1):
            k_w = float(conductivity[j, i])
            k_e = float(conductivity[j, i + 1])
            g = internal_conductance(
                grid.x_face_area, 0.5 * grid.dx, k_w, 0.5 * grid.dx, k_e
            )
            t_w = float(temperature[j, i])
            t_e = float(temperature[j, i + 1])
            flux = g * (t_w - t_e)
            weighted_w = (k_w / (0.5 * grid.dx)) * t_w
            weighted_e = (k_e / (0.5 * grid.dx)) * t_e
            face_t = (weighted_w + weighted_e) / (
                k_w / (0.5 * grid.dx) + k_e / (0.5 * grid.dx)
            )
            faces.append(
                {
                    "face_type": "internal_x",
                    "i": i,
                    "j": j,
                    "x": (i + 1) * grid.dx,
                    "y": (j + 0.5) * grid.dy,
                    "area": grid.x_face_area,
                    "k_p": k_w,
                    "k_n": k_e,
                    "conductance": g,
                    "temperature_face": face_t,
                    "flux_positive_axis": flux,
                }
            )

    # Internal y-faces; positive flux means south -> north.
    for j in range(grid.ny - 1):
        for i in range(grid.nx):
            k_s = float(conductivity[j, i])
            k_n = float(conductivity[j + 1, i])
            g = internal_conductance(
                grid.y_face_area, 0.5 * grid.dy, k_s, 0.5 * grid.dy, k_n
            )
            t_s = float(temperature[j, i])
            t_n = float(temperature[j + 1, i])
            flux = g * (t_s - t_n)
            weighted_s = (k_s / (0.5 * grid.dy)) * t_s
            weighted_n = (k_n / (0.5 * grid.dy)) * t_n
            face_t = (weighted_s + weighted_n) / (
                k_s / (0.5 * grid.dy) + k_n / (0.5 * grid.dy)
            )
            faces.append(
                {
                    "face_type": "internal_y",
                    "i": i,
                    "j": j,
                    "x": (i + 0.5) * grid.dx,
                    "y": (j + 1) * grid.dy,
                    "area": grid.y_face_area,
                    "k_p": k_s,
                    "k_n": k_n,
                    "conductance": g,
                    "temperature_face": face_t,
                    "flux_positive_axis": flux,
                }
            )

    for j in range(grid.ny):
        k_left = float(conductivity[j, 0])
        g_left = boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, k_left
        )
        faces.append(
            {
                "face_type": "boundary_left",
                "i": -1,
                "j": j,
                "x": 0.0,
                "y": (j + 0.5) * grid.dy,
                "area": grid.x_face_area,
                "k_p": k_left,
                "k_n": "",
                "conductance": g_left,
                "temperature_face": t_hot,
                "flux_positive_axis": g_left * (t_hot - float(temperature[j, 0])),
            }
        )

        k_right = float(conductivity[j, -1])
        g_right = boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, k_right
        )
        faces.append(
            {
                "face_type": "boundary_right",
                "i": grid.nx,
                "j": j,
                "x": grid.lx,
                "y": (j + 0.5) * grid.dy,
                "area": grid.x_face_area,
                "k_p": k_right,
                "k_n": "",
                "conductance": g_right,
                "temperature_face": t_cold,
                "flux_positive_axis": g_right
                * (float(temperature[j, -1]) - t_cold),
            }
        )

    # Store the homogeneous-Neumann faces explicitly as zero-flux evidence.
    for i in range(grid.nx):
        faces.append(
            {
                "face_type": "boundary_bottom",
                "i": i,
                "j": -1,
                "x": (i + 0.5) * grid.dx,
                "y": 0.0,
                "area": grid.y_face_area,
                "k_p": float(conductivity[0, i]),
                "k_n": "",
                "conductance": 0.0,
                "temperature_face": "",
                "flux_positive_axis": 0.0,
            }
        )
        faces.append(
            {
                "face_type": "boundary_top",
                "i": i,
                "j": grid.ny,
                "x": (i + 0.5) * grid.dx,
                "y": grid.ly,
                "area": grid.y_face_area,
                "k_p": float(conductivity[-1, i]),
                "k_n": "",
                "conductance": 0.0,
                "temperature_face": "",
                "flux_positive_axis": 0.0,
            }
        )

    return faces


def solve_grid(config: dict[str, Any], nx: int, ny: int) -> SolveResult:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")

    grid = Grid(
        nx=int(nx),
        ny=int(ny),
        lx=finite_positive(config["lx"], "lx"),
        ly=finite_positive(config["ly"], "ly"),
        thickness=finite_positive(config["thickness"], "thickness"),
    )
    conductivity = conductivity_field(config["case"], grid)
    matrix, rhs = assemble_system(config, grid, conductivity)
    solution = spsolve(matrix, rhs)
    if (
        solution.shape != (grid.nx * grid.ny,)
        or not np.all(np.isfinite(solution))
    ):
        raise RuntimeError("sparse solve returned invalid temperatures")

    residual = matrix @ solution - rhs
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    rhs_scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
    residual_rel_inf = residual_inf / rhs_scale
    temperature = solution.reshape((grid.ny, grid.nx))
    faces = reconstruct_faces(config, grid, temperature, conductivity)

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

    return SolveResult(
        grid=grid,
        temperature=temperature,
        conductivity=conductivity,
        residual_inf=residual_inf,
        residual_rel_inf=residual_rel_inf,
        q_hot=q_hot,
        q_cold=q_cold,
        faces=faces,
    )


def uniform_reference(
    config: dict[str, Any], grid: Grid
) -> tuple[np.ndarray, float]:
    k = float(config["case"]["k"])
    t_hot = float(config["t_hot"])
    t_cold = float(config["t_cold"])
    x = (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    exact_x = t_hot + (t_cold - t_hot) * x / grid.lx
    exact = np.tile(exact_x, (grid.ny, 1))
    area = grid.ly * grid.thickness
    q = k * area * (t_hot - t_cold) / grid.lx
    return exact, q


def layered_reference(
    config: dict[str, Any], grid: Grid
) -> tuple[np.ndarray, float, float]:
    case = config["case"]
    x_int = float(case["interface_x"])
    k_left = float(case["k_left"])
    k_right = float(case["k_right"])
    t_hot = float(config["t_hot"])
    t_cold = float(config["t_cold"])
    area = grid.ly * grid.thickness

    r_left = x_int / (k_left * area)
    r_right = (grid.lx - x_int) / (k_right * area)
    q_total = (t_hot - t_cold) / (r_left + r_right)
    t_interface = t_hot - q_total * r_left
    q_density = q_total / area

    x = (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    exact_x = np.empty(grid.nx, dtype=float)
    for i, xc in enumerate(x):
        if xc < x_int:
            exact_x[i] = t_hot - q_density * xc / k_left
        else:
            exact_x[i] = (
                t_interface - q_density * (xc - x_int) / k_right
            )
    exact = np.tile(exact_x, (grid.ny, 1))
    return exact, q_total, t_interface


def summarize_result(
    config: dict[str, Any], result: SolveResult
) -> tuple[dict[str, Any], np.ndarray]:
    kind = str(config["case"]["kind"])
    left_extrapolated = ""
    right_extrapolated = ""

    if kind == "uniform":
        exact, q_exact = uniform_reference(config, result.grid)
        t_interface_exact = ""
        interface_face_temp_error = ""
        interface_flux_rel_error = ""
    elif kind == "layered":
        exact, q_exact, t_interface = layered_reference(config, result.grid)
        t_interface_exact = t_interface
        x_int = float(config["case"]["interface_x"])
        interface_faces = [
            face
            for face in result.faces
            if face["face_type"] == "internal_x"
            and abs(float(face["x"]) - x_int)
            <= 5e-12 * max(1.0, result.grid.lx)
        ]
        if len(interface_faces) != result.grid.ny:
            raise RuntimeError(
                f"expected {result.grid.ny} interface faces, "
                f"got {len(interface_faces)}"
            )
        interface_face_temp_error = max(
            abs(float(face["temperature_face"]) - t_interface)
            for face in interface_faces
        )
        q_face_exact = q_exact / result.grid.ny
        interface_flux_rel_error = max(
            abs(float(face["flux_positive_axis"]) - q_face_exact)
            / abs(q_face_exact)
            for face in interface_faces
        )

        x_centers = (
            np.arange(result.grid.nx, dtype=float) + 0.5
        ) * result.grid.dx
        mean_profile = np.mean(result.temperature, axis=0)
        left_mask = x_centers < x_int
        right_mask = x_centers > x_int
        left_coeff = np.polyfit(
            x_centers[left_mask], mean_profile[left_mask], 1
        )
        right_coeff = np.polyfit(
            x_centers[right_mask], mean_profile[right_mask], 1
        )
        left_extrapolated = float(np.polyval(left_coeff, x_int))
        right_extrapolated = float(np.polyval(right_coeff, x_int))
    else:
        raise ValueError(kind)

    error = result.temperature - exact
    max_temp_error = float(np.max(np.abs(error)))
    q_hot_rel_error = abs(result.q_hot - q_exact) / abs(q_exact)
    q_cold_rel_error = abs(result.q_cold - q_exact) / abs(q_exact)
    balance_rel = abs(result.q_hot - result.q_cold) / max(
        abs(result.q_hot), abs(result.q_cold), 1e-300
    )

    summary = {
        "experiment_id": config["experiment_id"],
        "case_name": config["case_name"],
        "case_kind": kind,
        "nx": result.grid.nx,
        "ny": result.grid.ny,
        "dx": result.grid.dx,
        "dy": result.grid.dy,
        "unknowns": result.grid.nx * result.grid.ny,
        "temperature_min": float(np.min(result.temperature)),
        "temperature_max": float(np.max(result.temperature)),
        "max_temperature_error": max_temp_error,
        "q_hot": result.q_hot,
        "q_cold": result.q_cold,
        "q_exact": q_exact,
        "q_hot_relative_error": q_hot_rel_error,
        "q_cold_relative_error": q_cold_rel_error,
        "global_balance_relative": balance_rel,
        "residual_inf": result.residual_inf,
        "residual_relative_inf": result.residual_rel_inf,
        "interface_temperature_exact": t_interface_exact,
        "interface_face_temperature_max_error": interface_face_temp_error,
        "interface_face_flux_max_relative_error": interface_flux_rel_error,
        "interface_left_extrapolated_temperature": left_extrapolated,
        "interface_right_extrapolated_temperature": right_extrapolated,
        "interface_left_extrapolation_error": (
            abs(float(left_extrapolated) - float(t_interface_exact))
            if kind == "layered"
            else ""
        ),
        "interface_right_extrapolation_error": (
            abs(float(right_extrapolated) - float(t_interface_exact))
            if kind == "layered"
            else ""
        ),
    }
    return summary, exact


def verify_summary(
    config: dict[str, Any], summary: dict[str, Any]
) -> None:
    if float(summary["max_temperature_error"]) > 1e-9:
        raise AssertionError(
            f"{config['experiment_id']} temperature error exceeds 1e-9 K"
        )
    if float(summary["q_hot_relative_error"]) > 1e-9:
        raise AssertionError(
            f"{config['experiment_id']} hot-boundary heat-rate error "
            "exceeds 1e-9"
        )
    if float(summary["q_cold_relative_error"]) > 1e-9:
        raise AssertionError(
            f"{config['experiment_id']} cold-boundary heat-rate error "
            "exceeds 1e-9"
        )
    if float(summary["global_balance_relative"]) > 1e-10:
        raise AssertionError(
            f"{config['experiment_id']} global heat balance exceeds 1e-10"
        )
    if float(summary["residual_relative_inf"]) > 1e-11:
        raise AssertionError(
            f"{config['experiment_id']} relative residual exceeds 1e-11"
        )
    if config["experiment_id"] == "ET001-02":
        if float(summary["interface_face_temperature_max_error"]) > 1e-9:
            raise AssertionError(
                "ET001-02 interface temperature error exceeds 1e-9 K"
            )
        if float(summary["interface_face_flux_max_relative_error"]) > 1e-9:
            raise AssertionError(
                "ET001-02 interface face flux error exceeds 1e-9"
            )
        if float(summary["interface_left_extrapolation_error"]) > 1e-9:
            raise AssertionError(
                "ET001-02 left extrapolated interface temperature "
                "exceeds 1e-9 K"
            )
        if float(summary["interface_right_extrapolation_error"]) > 1e-9:
            raise AssertionError(
                "ET001-02 right extrapolated interface temperature "
                "exceeds 1e-9 K"
            )


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


def run_config(
    config_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "experiment_id",
        "case_name",
        "lx",
        "ly",
        "thickness",
        "t_hot",
        "t_cold",
        "case",
        "grids",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config fields: {missing}")

    if config["experiment_id"] not in {"ET001-01", "ET001-02"}:
        raise ValueError(
            "this implementation intentionally supports only "
            "ET001-01 and ET001-02"
        )

    summaries: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    face_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for grid_pair in config["grids"]:
        nx, ny = int(grid_pair[0]), int(grid_pair[1])
        result = solve_grid(config, nx, ny)
        summary, exact = summarize_result(config, result)
        verify_summary(config, summary)
        summaries.append(summary)

        for j in range(result.grid.ny):
            for i in range(result.grid.nx):
                cell_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "case_name": config["case_name"],
                        "nx": nx,
                        "ny": ny,
                        "i": i,
                        "j": j,
                        "x": (i + 0.5) * result.grid.dx,
                        "y": (j + 0.5) * result.grid.dy,
                        "k": float(result.conductivity[j, i]),
                        "temperature": float(result.temperature[j, i]),
                        "temperature_exact": float(exact[j, i]),
                        "error": float(
                            result.temperature[j, i] - exact[j, i]
                        ),
                    }
                )

        if config["experiment_id"] == "ET001-02":
            for face in result.faces:
                face_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "case_name": config["case_name"],
                        "nx": nx,
                        "ny": ny,
                        **face,
                    }
                )

            for i in range(result.grid.nx):
                profile_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "case_name": config["case_name"],
                        "nx": nx,
                        "ny": ny,
                        "x": (i + 0.5) * result.grid.dx,
                        "temperature_mean_y": float(
                            np.mean(result.temperature[:, i])
                        ),
                        "temperature_exact": float(exact[0, i]),
                        "error": float(
                            np.mean(result.temperature[:, i]) - exact[0, i]
                        ),
                        "material": (
                            "left"
                            if (i + 0.5) * result.grid.dx
                            < float(config["case"]["interface_x"])
                            else "right"
                        ),
                    }
                )

    summary_fields = [
        "experiment_id",
        "case_name",
        "case_kind",
        "nx",
        "ny",
        "dx",
        "dy",
        "unknowns",
        "temperature_min",
        "temperature_max",
        "max_temperature_error",
        "q_hot",
        "q_cold",
        "q_exact",
        "q_hot_relative_error",
        "q_cold_relative_error",
        "global_balance_relative",
        "residual_inf",
        "residual_relative_inf",
        "interface_temperature_exact",
        "interface_face_temperature_max_error",
        "interface_face_flux_max_relative_error",
        "interface_left_extrapolated_temperature",
        "interface_right_extrapolated_temperature",
        "interface_left_extrapolation_error",
        "interface_right_extrapolation_error",
    ]
    write_csv(output_dir / "summary.csv", summaries, summary_fields)

    cell_fields = [
        "experiment_id",
        "case_name",
        "nx",
        "ny",
        "i",
        "j",
        "x",
        "y",
        "k",
        "temperature",
        "temperature_exact",
        "error",
    ]
    write_csv(output_dir / "cells.csv", cell_rows, cell_fields)

    if face_rows:
        face_fields = [
            "experiment_id",
            "case_name",
            "nx",
            "ny",
            "face_type",
            "i",
            "j",
            "x",
            "y",
            "area",
            "k_p",
            "k_n",
            "conductance",
            "temperature_face",
            "flux_positive_axis",
        ]
        write_csv(output_dir / "faces.csv", face_rows, face_fields)

    if profile_rows:
        profile_fields = [
            "experiment_id",
            "case_name",
            "nx",
            "ny",
            "x",
            "temperature_mean_y",
            "temperature_exact",
            "error",
            "material",
        ]
        write_csv(
            output_dir / "layered_profile.csv",
            profile_rows,
            profile_fields,
        )

    run_lines = [
        f"experiment_id={config['experiment_id']}",
        f"case_name={config['case_name']}",
        f"config={config_path.as_posix()}",
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"platform={platform.platform()}",
        "solver=scipy.sparse.linalg.spsolve",
        (
            "face_scheme=cell-centered TPFA with distance-weighted "
            "harmonic conductance"
        ),
        "top_bottom_boundary=homogeneous Neumann",
        (
            "left_right_boundary=Dirichlet with center-to-face "
            "half-cell distance"
        ),
        "status=PASS",
    ]
    for summary in summaries:
        run_lines.append(
            "grid="
            f"{summary['nx']}x{summary['ny']} "
            f"max_temperature_error="
            f"{float(summary['max_temperature_error']):.17g} "
            f"q_hot={float(summary['q_hot']):.17g} "
            f"q_cold={float(summary['q_cold']):.17g} "
            f"q_exact={float(summary['q_exact']):.17g} "
            f"balance_relative="
            f"{float(summary['global_balance_relative']):.17g} "
            f"residual_relative_inf="
            f"{float(summary['residual_relative_inf']):.17g}"
        )
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
    summaries = run_config(args.config, args.output)
    for summary in summaries:
        print(
            f"{summary['experiment_id']} "
            f"{summary['nx']}x{summary['ny']} "
            f"max_T_error="
            f"{float(summary['max_temperature_error']):.3e} "
            f"q_hot={float(summary['q_hot']):.12g} "
            f"q_exact={float(summary['q_exact']):.12g} "
            f"balance_rel="
            f"{float(summary['global_balance_relative']):.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
