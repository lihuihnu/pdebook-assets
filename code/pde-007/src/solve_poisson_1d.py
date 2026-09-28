#!/usr/bin/env python3
"""Solve the pde-007 N=4 one-dimensional Poisson teaching example.

The script uses only the Python standard library. It solves the same 3x3
tridiagonal system derived in the article by plain Gaussian elimination.
This is a teaching implementation, not a solver recommendation for large
PDE systems.
"""

import csv
from pathlib import Path


N = 4
h = 1.0 / N
tol = 1.0e-12


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
    return x * (1.0 - x)


def solve_problem():
    nodes = []
    for i in range(N + 1):
        nodes.append(i * h)

    A = [
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0],
    ]
    b = [2.0 * h * h, 2.0 * h * h, 2.0 * h * h]
    interior_solution = gaussian_elimination(A, b)

    U = [0.0] * (N + 1)
    for i in range(1, N):
        U[i] = interior_solution[i - 1]
    return nodes, A, b, U


def main():
    nodes, A, b, U = solve_problem()

    expected_interior = [3.0 / 16.0, 1.0 / 4.0, 3.0 / 16.0]
    for i in range(N - 1):
        assert abs(U[i + 1] - expected_interior[i]) < tol
    assert abs(U[0]) < tol
    assert abs(U[N]) < tol
    for i in range(1, N):
        lhs = -U[i - 1] + 2.0 * U[i] - U[i + 1]
        assert abs(lhs - 2.0 * h * h) < tol
    for i in range(N + 1):
        assert abs(U[i] - exact_solution(nodes[i])) < tol

    article_root = Path(__file__).resolve().parents[1]
    output = article_root / "results" / "n4_solution.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i", "x", "U", "exact"])
        for i in range(N + 1):
            writer.writerow(
                [i, f"{nodes[i]:.12g}", f"{U[i]:.12g}", f"{exact_solution(nodes[i]):.12g}"]
            )

    print("i    x       U        exact")
    for i in range(N + 1):
        print(f"{i}    {nodes[i]:.2f}    {U[i]:.6f}    {exact_solution(nodes[i]):.6f}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
