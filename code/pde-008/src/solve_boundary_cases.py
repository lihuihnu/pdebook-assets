#!/usr/bin/env python3
"""Solve the pde-008 N=4 boundary-condition teaching examples.

The script uses only the Python standard library. It solves three small
finite-difference systems for the same one-dimensional Poisson equation,
changing only the right boundary condition: Dirichlet, Neumann, or Robin.
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
    return 1.0 + x - x * x


def build_systems():
    dirichlet_A = [
        [2.0, -1.0, 0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0, 2.0],
    ]
    dirichlet_b = [9.0 / 8.0, 1.0 / 8.0, 9.0 / 8.0]

    neumann_A = [
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 1.0, -4.0, 3.0],
    ]
    neumann_b = [9.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, -1.0 / 2.0]

    robin_A = [
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 1.0, -4.0, 7.0 / 2.0],
    ]
    robin_b = [9.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 0.0]

    return {
        "dirichlet": (dirichlet_A, dirichlet_b),
        "neumann": (neumann_A, neumann_b),
        "robin": (robin_A, robin_b),
    }


def build_full_solution(name, solution):
    if name == "dirichlet":
        return [1.0] + solution + [1.0]
    return [1.0] + solution


def check_internal_equations(U):
    for i in range(1, N):
        lhs = -U[i - 1] + 2.0 * U[i] - U[i + 1]
        assert abs(lhs - 2.0 * h * h) < tol


def check_exact_nodes(nodes, U):
    for i in range(N + 1):
        assert abs(U[i] - exact_solution(nodes[i])) < tol


def main():
    nodes = []
    for i in range(N + 1):
        nodes.append(i * h)

    systems = build_systems()
    solutions = {}

    for name, (A, b) in systems.items():
        solution = gaussian_elimination(A, b)
        U = build_full_solution(name, solution)
        solutions[name] = U

    expected = [1.0, 19.0 / 16.0, 5.0 / 4.0, 19.0 / 16.0, 1.0]

    for name, U in solutions.items():
        for i in range(N + 1):
            assert abs(U[i] - expected[i]) < tol
        check_internal_equations(U)
        check_exact_nodes(nodes, U)

        if name == "dirichlet":
            assert abs(U[0] - 1.0) < tol
            assert abs(U[N] - 1.0) < tol
        elif name == "neumann":
            derivative = (3.0 * U[N] - 4.0 * U[N - 1] + U[N - 2]) / (2.0 * h)
            assert abs(derivative + 1.0) < tol
        elif name == "robin":
            derivative = (3.0 * U[N] - 4.0 * U[N - 1] + U[N - 2]) / (2.0 * h)
            assert abs(derivative + U[N]) < tol

    baseline = solutions["dirichlet"]
    for name in ("neumann", "robin"):
        for i in range(N + 1):
            assert abs(solutions[name][i] - baseline[i]) < tol

    article_root = Path(__file__).resolve().parents[1]
    output = article_root / "results" / "boundary_cases.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "i", "x", "U", "exact"])
        for name in ("dirichlet", "neumann", "robin"):
            U = solutions[name]
            for i in range(N + 1):
                writer.writerow(
                    [
                        name,
                        i,
                        f"{nodes[i]:.12g}",
                        f"{U[i]:.12g}",
                        f"{exact_solution(nodes[i]):.12g}",
                    ]
                )

    print("case        i    x       U        exact")
    for name in ("dirichlet", "neumann", "robin"):
        U = solutions[name]
        for i in range(N + 1):
            print(
                f"{name:10s}  {i}    {nodes[i]:.2f}    "
                f"{U[i]:.6f}    {exact_solution(nodes[i]):.6f}"
            )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
