#!/usr/bin/env python3
"""Run the pde-011 Jacobi and Gauss-Seidel teaching experiment.

The experiment reuses the fixed 9x9 linear system from pde-010. It starts both
iterations from the zero vector and records the infinity norm of the residual,
the infinity norm of the iteration error, and the center-node value after every
completed iteration.

Only the Python standard library is used.
"""

import csv
from pathlib import Path


TOLERANCE = 1.0e-10
MAX_ITERATIONS = 10000
SIZE = 9
CENTER_INDEX = 4


def build_system():
    """Return the fixed pde-010 matrix, right-hand side, and discrete solution."""
    A = [
        [4.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 4.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, -1.0, 4.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 4.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 4.0],
    ]
    b = [
        6.0 / 128.0,
        7.0 / 128.0,
        6.0 / 128.0,
        7.0 / 128.0,
        8.0 / 128.0,
        7.0 / 128.0,
        6.0 / 128.0,
        7.0 / 128.0,
        6.0 / 128.0,
    ]
    exact = [
        9.0 / 256.0,
        12.0 / 256.0,
        9.0 / 256.0,
        12.0 / 256.0,
        16.0 / 256.0,
        12.0 / 256.0,
        9.0 / 256.0,
        12.0 / 256.0,
        9.0 / 256.0,
    ]
    return A, b, exact


def residual_inf(A, b, U):
    """Return max_k |b_k - (A U)_k|."""
    maximum = 0.0
    for row in range(SIZE):
        value = 0.0
        for col in range(SIZE):
            value += A[row][col] * U[col]
        maximum = max(maximum, abs(b[row] - value))
    return maximum


def iteration_error_inf(U, exact):
    """Return max_k |U_k - U_{h,k}| for the known teaching solution."""
    maximum = 0.0
    for k in range(SIZE):
        maximum = max(maximum, abs(U[k] - exact[k]))
    return maximum


def record(method, iteration, A, b, U, exact):
    """Build one transparent iteration-history row."""
    return {
        "method": method,
        "iteration": iteration,
        "residual_inf": residual_inf(A, b, U),
        "iteration_error_inf": iteration_error_inf(U, exact),
        "center_value": U[CENTER_INDEX],
    }


def jacobi(A, b, exact):
    """Run Jacobi from the zero vector until the residual tolerance is met."""
    U = [0.0] * SIZE
    history = [record("jacobi", 0, A, b, U, exact)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_U = [0.0] * SIZE
        for row in range(SIZE):
            off_diagonal = 0.0
            for col in range(SIZE):
                if col != row:
                    off_diagonal += A[row][col] * U[col]
            new_U[row] = (b[row] - off_diagonal) / A[row][row]

        U = new_U
        history.append(record("jacobi", iteration, A, b, U, exact))
        if history[-1]["residual_inf"] < TOLERANCE:
            return U, history

    raise RuntimeError("Jacobi did not reach the residual tolerance")


def gauss_seidel(A, b, exact):
    """Run Gauss-Seidel from the zero vector in the pde-010 row-major order."""
    U = [0.0] * SIZE
    history = [record("gauss_seidel", 0, A, b, U, exact)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        for row in range(SIZE):
            lower = 0.0
            for col in range(row):
                lower += A[row][col] * U[col]

            upper = 0.0
            for col in range(row + 1, SIZE):
                upper += A[row][col] * U[col]

            U[row] = (b[row] - lower - upper) / A[row][row]

        history.append(record("gauss_seidel", iteration, A, b, U, exact))
        if history[-1]["residual_inf"] < TOLERANCE:
            return U[:], history

    raise RuntimeError("Gauss-Seidel did not reach the residual tolerance")


def grid_indices(k):
    """Map zero-based vector position k to the one-based pde-010 (i, j)."""
    n = 3
    i = k % n + 1
    j = k // n + 1
    return i, j


def write_history(path, histories):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "iteration",
                "residual_inf",
                "iteration_error_inf",
                "center_value",
            ]
        )
        for history in histories:
            for row in history:
                writer.writerow(
                    [
                        row["method"],
                        row["iteration"],
                        f'{row["residual_inf"]:.12g}',
                        f'{row["iteration_error_inf"]:.12g}',
                        f'{row["center_value"]:.12g}',
                    ]
                )


def write_final_solutions(path, solutions, exact, histories):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "iterations", "k", "i", "j", "U", "exact", "error"])
        for method, U in solutions:
            iterations = len(histories[method]) - 1
            for k in range(SIZE):
                i, j = grid_indices(k)
                writer.writerow(
                    [
                        method,
                        iterations,
                        k + 1,
                        i,
                        j,
                        f"{U[k]:.12g}",
                        f"{exact[k]:.12g}",
                        f"{U[k] - exact[k]:.12g}",
                    ]
                )


def main():
    A, b, exact = build_system()

    jacobi_solution, jacobi_history = jacobi(A, b, exact)
    gs_solution, gs_history = gauss_seidel(A, b, exact)

    # Contract checks against pde-010 and the pde-011 hand calculations.
    assert abs(b[4] - 1.0 / 16.0) < 1.0e-15
    assert abs(exact[4] - 1.0 / 16.0) < 1.0e-15
    assert abs(jacobi_history[1]["center_value"] - 1.0 / 64.0) < 1.0e-15
    assert abs(jacobi_history[2]["center_value"] - 15.0 / 512.0) < 1.0e-15
    assert abs(jacobi_history[0]["residual_inf"] - 1.0 / 16.0) < 1.0e-15

    assert jacobi_history[-1]["residual_inf"] < TOLERANCE
    assert gs_history[-1]["residual_inf"] < TOLERANCE
    assert len(jacobi_history) - 1 == 60
    assert len(gs_history) - 1 == 31

    article_root = Path(__file__).resolve().parents[1]
    results_dir = article_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history_output = results_dir / "iteration_history.csv"
    final_output = results_dir / "final_solutions.csv"

    write_history(history_output, [jacobi_history, gs_history])
    histories = {
        "jacobi": jacobi_history,
        "gauss_seidel": gs_history,
    }
    write_final_solutions(
        final_output,
        [("jacobi", jacobi_solution), ("gauss_seidel", gs_solution)],
        exact,
        histories,
    )

    print(f"tolerance = {TOLERANCE:.1e}")
    print(
        "Jacobi: "
        f"iterations={len(jacobi_history) - 1}, "
        f"residual_inf={jacobi_history[-1]['residual_inf']:.3e}, "
        f"iteration_error_inf={jacobi_history[-1]['iteration_error_inf']:.3e}"
    )
    print(
        "Gauss-Seidel: "
        f"iterations={len(gs_history) - 1}, "
        f"residual_inf={gs_history[-1]['residual_inf']:.3e}, "
        f"iteration_error_inf={gs_history[-1]['iteration_error_inf']:.3e}"
    )
    print(f"wrote {history_output}")
    print(f"wrote {final_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
