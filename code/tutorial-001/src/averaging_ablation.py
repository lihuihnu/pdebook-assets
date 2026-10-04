#!/usr/bin/env python3
"""ET001-03: harmonic-versus-arithmetic interface averaging ablation.

The verified harmonic solver in composite_heat.py is treated as frozen.
This script imports that path unchanged for the harmonic rows and implements a
separate, deliberately non-default arithmetic face-coefficient path for the
ablation only.
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
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import composite_heat as core


FROZEN_CORE_BLOB = "be4d7e82ce473c0259017813a2b2b5f580d891fa"


def arithmetic_internal_conductance(
    area: float,
    distance_p: float,
    k_p: float,
    distance_n: float,
    k_n: float,
) -> float:
    """Deliberate ET001-03 ablation: arithmetic face coefficient."""
    distance = distance_p + distance_n
    k_face = 0.5 * (k_p + k_n)
    return k_face * area / distance


def assemble_arithmetic_system(
    config: dict[str, Any],
    grid: core.Grid,
    conductivity: np.ndarray,
):
    unknown_count = grid.nx * grid.ny
    matrix = lil_matrix((unknown_count, unknown_count), dtype=float)
    rhs = np.zeros(unknown_count, dtype=float)
    qv = core.finite_value(config.get("qv", 0.0), "qv")
    t_hot = core.finite_value(config["t_hot"], "t_hot")
    t_cold = core.finite_value(config["t_cold"], "t_cold")
    rhs[:] = qv * grid.cell_volume

    for j in range(grid.ny):
        for i in range(grid.nx - 1):
            p = core.cell_index(i, j, grid.nx)
            n = core.cell_index(i + 1, j, grid.nx)
            g = arithmetic_internal_conductance(
                grid.x_face_area,
                0.5 * grid.dx,
                float(conductivity[j, i]),
                0.5 * grid.dx,
                float(conductivity[j, i + 1]),
            )
            matrix[p, p] += g
            matrix[n, n] += g
            matrix[p, n] -= g
            matrix[n, p] -= g

    for j in range(grid.ny - 1):
        for i in range(grid.nx):
            p = core.cell_index(i, j, grid.nx)
            n = core.cell_index(i, j + 1, grid.nx)
            g = arithmetic_internal_conductance(
                grid.y_face_area,
                0.5 * grid.dy,
                float(conductivity[j, i]),
                0.5 * grid.dy,
                float(conductivity[j + 1, i]),
            )
            matrix[p, p] += g
            matrix[n, n] += g
            matrix[p, n] -= g
            matrix[n, p] -= g

    for j in range(grid.ny):
        p_left = core.cell_index(0, j, grid.nx)
        g_left = core.boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, float(conductivity[j, 0])
        )
        matrix[p_left, p_left] += g_left
        rhs[p_left] += g_left * t_hot

        p_right = core.cell_index(grid.nx - 1, j, grid.nx)
        g_right = core.boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, float(conductivity[j, -1])
        )
        matrix[p_right, p_right] += g_right
        rhs[p_right] += g_right * t_cold

    return matrix.tocsr(), rhs


def arithmetic_discrete_heat_rate(
    config: dict[str, Any],
    grid: core.Grid,
) -> float:
    """Independent 1D series resistance of the arithmetic discrete stencil."""
    case = config["case"]
    k_left = float(case["k_left"])
    k_right = float(case["k_right"])
    x_int = float(case["interface_x"])
    area = grid.ly * grid.thickness
    dx = grid.dx
    k_face = 0.5 * (k_left + k_right)

    resistance = (
        (x_int - 0.5 * dx) / (k_left * area)
        + dx / (k_face * area)
        + (grid.lx - x_int - 0.5 * dx) / (k_right * area)
    )
    return (float(config["t_hot"]) - float(config["t_cold"])) / resistance


def solve_arithmetic(
    config: dict[str, Any],
    nx: int,
    ny: int,
) -> dict[str, Any]:
    grid = core.Grid(
        nx=int(nx),
        ny=int(ny),
        lx=core.finite_positive(config["lx"], "lx"),
        ly=core.finite_positive(config["ly"], "ly"),
        thickness=core.finite_positive(config["thickness"], "thickness"),
    )
    conductivity = core.conductivity_field(config["case"], grid)
    matrix, rhs = assemble_arithmetic_system(config, grid, conductivity)
    solution = spsolve(matrix, rhs)
    if (
        solution.shape != (grid.nx * grid.ny,)
        or not np.all(np.isfinite(solution))
    ):
        raise RuntimeError("arithmetic ablation solve returned invalid values")

    temperature = solution.reshape((grid.ny, grid.nx))
    residual = matrix @ solution - rhs
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    rhs_scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
    residual_relative_inf = residual_inf / rhs_scale

    t_hot = float(config["t_hot"])
    t_cold = float(config["t_cold"])
    q_hot = 0.0
    q_cold = 0.0
    for j in range(grid.ny):
        g_left = core.boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, float(conductivity[j, 0])
        )
        q_hot += g_left * (t_hot - float(temperature[j, 0]))

        g_right = core.boundary_conductance(
            grid.x_face_area, 0.5 * grid.dx, float(conductivity[j, -1])
        )
        q_cold += g_right * (float(temperature[j, -1]) - t_cold)

    exact_temperature, q_exact, _ = core.layered_reference(config, grid)
    q_discrete_prediction = arithmetic_discrete_heat_rate(config, grid)
    max_temperature_error = float(
        np.max(np.abs(temperature - exact_temperature))
    )
    balance_relative = abs(q_hot - q_cold) / max(
        abs(q_hot), abs(q_cold), 1e-300
    )

    return {
        "scheme": "arithmetic_ablation",
        "nx": grid.nx,
        "ny": grid.ny,
        "dx": grid.dx,
        "dy": grid.dy,
        "q_hot": q_hot,
        "q_cold": q_cold,
        "q_exact_physical": q_exact,
        "q_discrete_prediction": q_discrete_prediction,
        "q_hot_relative_error_physical": abs(q_hot - q_exact) / abs(q_exact),
        "q_hot_relative_error_discrete_prediction": (
            abs(q_hot - q_discrete_prediction)
            / abs(q_discrete_prediction)
        ),
        "max_temperature_error_physical": max_temperature_error,
        "global_balance_relative": balance_relative,
        "residual_inf": residual_inf,
        "residual_relative_inf": residual_relative_inf,
    }


def solve_harmonic(
    config: dict[str, Any],
    nx: int,
    ny: int,
) -> dict[str, Any]:
    result = core.solve_grid(config, nx, ny)
    exact_temperature, q_exact, _ = core.layered_reference(config, result.grid)
    max_temperature_error = float(
        np.max(np.abs(result.temperature - exact_temperature))
    )
    return {
        "scheme": "harmonic_frozen",
        "nx": result.grid.nx,
        "ny": result.grid.ny,
        "dx": result.grid.dx,
        "dy": result.grid.dy,
        "q_hot": result.q_hot,
        "q_cold": result.q_cold,
        "q_exact_physical": q_exact,
        "q_discrete_prediction": q_exact,
        "q_hot_relative_error_physical": abs(result.q_hot - q_exact)
        / abs(q_exact),
        "q_hot_relative_error_discrete_prediction": abs(
            result.q_hot - q_exact
        )
        / abs(q_exact),
        "max_temperature_error_physical": max_temperature_error,
        "global_balance_relative": abs(result.q_hot - result.q_cold)
        / max(abs(result.q_hot), abs(result.q_cold), 1e-300),
        "residual_inf": result.residual_inf,
        "residual_relative_inf": result.residual_rel_inf,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no ablation rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "ET001-03":
        raise ValueError("averaging_ablation.py only accepts ET001-03")

    contrasts = [core.finite_positive(v, "contrast") for v in config["contrasts"]]
    grids = [(int(v[0]), int(v[1])) for v in config["grids"]]
    k_left = core.finite_positive(config["k_left"], "k_left")

    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        layered_config = {
            "experiment_id": "ET001-03",
            "case_name": f"layered_contrast_{contrast:g}",
            "lx": config["lx"],
            "ly": config["ly"],
            "thickness": config["thickness"],
            "t_hot": config["t_hot"],
            "t_cold": config["t_cold"],
            "qv": 0.0,
            "case": {
                "kind": "layered",
                "interface_x": config["interface_x"],
                "k_left": k_left,
                "k_right": k_left * contrast,
            },
        }
        for nx, ny in grids:
            for result in (
                solve_harmonic(layered_config, nx, ny),
                solve_arithmetic(layered_config, nx, ny),
            ):
                rows.append(
                    {
                        "experiment_id": "ET001-03",
                        "contrast": contrast,
                        "k_left": k_left,
                        "k_right": k_left * contrast,
                        **result,
                    }
                )

    write_csv(output_dir / "averaging_ablation.csv", rows)

    run_lines = [
        "experiment_id=ET001-03",
        f"config={config_path.as_posix()}",
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"platform={platform.platform()}",
        f"frozen_core_blob={FROZEN_CORE_BLOB}",
        "harmonic_path=imported unchanged from composite_heat.py",
        "arithmetic_path=deliberate non-default ablation",
        "status=PASS",
    ]
    for row in rows:
        run_lines.append(
            f"contrast={float(row['contrast']):g} "
            f"scheme={row['scheme']} "
            f"grid={row['nx']}x{row['ny']} "
            f"q_hot={float(row['q_hot']):.17g} "
            f"q_exact={float(row['q_exact_physical']):.17g} "
            f"rel_error_physical="
            f"{float(row['q_hot_relative_error_physical']):.17g} "
            f"rel_error_discrete_prediction="
            f"{float(row['q_hot_relative_error_discrete_prediction']):.17g}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.txt").write_text(
        "\n".join(run_lines) + "\n", encoding="utf-8"
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = run(args.config, args.output)
    for row in rows:
        print(
            f"chi={float(row['contrast']):g} "
            f"{row['scheme']} {row['nx']}x{row['ny']} "
            f"q={float(row['q_hot']):.12g} "
            f"physical_rel_error="
            f"{float(row['q_hot_relative_error_physical']):.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
