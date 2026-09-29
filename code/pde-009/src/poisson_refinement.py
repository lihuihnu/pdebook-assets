#!/usr/bin/env python3
"""Run the pde-009 one-dimensional Poisson grid-refinement experiment.

The experiment solves

    -u''(x) = 12 x^2,  0 < x < 1,
    u(0) = u(1) = 0,

whose exact solution is u(x) = x - x^4. It uses only the Python standard
library and the same small-system Gaussian elimination idea introduced in
pde-007. The goal is to measure nodal error under grid refinement, not to
recommend a large-scale linear solver.
"""

import csv
import math
from pathlib import Path


GRID_SIZES = [4, 8, 16, 32, 64]
tol = 1.0e-10


def gaussian_elimination(A, b):
    """Solve a small nonsingular linear system by direct elimination."""
    n = len(b)
    augmented = []
    for i in range(n):
        augmented.append(A[i][:] + [b[i]])

    for k in range(n - 1):
        pivot = augmented[k][k]
        if abs(pivot) < tol:
            raise ValueError("zero or near-zero pivot in teaching solver")
        for i in range(k + 1, n):
            factor = augmented[i][k] / pivot
            if factor == 0.0:
                continue
            for j in range(k, n + 1):
                augmented[i][j] -= factor * augmented[k][j]

    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        value = augmented[i][n]
        for j in range(i + 1, n):
            value -= augmented[i][j] * solution[j]
        pivot = augmented[i][i]
        if abs(pivot) < tol:
            raise ValueError("zero or near-zero pivot in teaching solver")
        solution[i] = value / pivot
    return solution


def exact_solution(x):
    return x - x**4


def rhs(x):
    return 12.0 * x * x


def build_system(N):
    h = 1.0 / N
    interior_count = N - 1
    A = []
    b = []

    for row in range(interior_count):
        i = row + 1
        x_i = i * h

        matrix_row = [0.0] * interior_count
        matrix_row[row] = 2.0
        if row > 0:
            matrix_row[row - 1] = -1.0
        if row < interior_count - 1:
            matrix_row[row + 1] = -1.0

        A.append(matrix_row)
        b.append(h * h * rhs(x_i))

    return h, A, b


def solve_on_grid(N):
    h, A, b = build_system(N)
    interior_solution = gaussian_elimination(A, b)

    nodes = []
    U = [0.0] * (N + 1)
    for i in range(N + 1):
        nodes.append(i * h)
    for i in range(1, N):
        U[i] = interior_solution[i - 1]

    errors = []
    for i in range(N + 1):
        errors.append(U[i] - exact_solution(nodes[i]))

    max_error = max(abs(error) for error in errors)

    for i in range(N + 1):
        expected_error = -h * h * nodes[i] * (1.0 - nodes[i])
        assert abs(errors[i] - expected_error) < tol
    assert abs(max_error - h * h / 4.0) < tol

    return h, max_error


def run_refinement():
    rows = []
    previous_error = None

    for N in GRID_SIZES:
        h, max_error = solve_on_grid(N)

        error_ratio = None
        observed_order = None
        if previous_error is not None:
            error_ratio = previous_error / max_error
            observed_order = math.log(error_ratio) / math.log(2.0)

        rows.append((N, h, max_error, error_ratio, observed_order))
        previous_error = max_error

    return rows


def main():
    rows = run_refinement()

    article_root = Path(__file__).resolve().parents[1]
    output = article_root / "results" / "convergence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "h", "max_error", "error_ratio", "observed_order"])
        for N, h, max_error, error_ratio, observed_order in rows:
            ratio = "" if error_ratio is None else f"{error_ratio:.6f}"
            order = "" if observed_order is None else f"{observed_order:.6f}"
            writer.writerow(
                [
                    N,
                    f"{h:.10g}",
                    f"{max_error:.10g}",
                    ratio,
                    order,
                ]
            )

    print("N    h           max_error       ratio       order")
    for N, h, max_error, error_ratio, observed_order in rows:
        ratio = "-" if error_ratio is None else f"{error_ratio:.6f}"
        order = "-" if observed_order is None else f"{observed_order:.6f}"
        print(
            f"{N:2d}   {h:.8f}   {max_error:.10f}   "
            f"{ratio:>10s}   {order:>8s}"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
