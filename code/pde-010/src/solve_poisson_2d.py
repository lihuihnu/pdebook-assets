#!/usr/bin/env python3
"""Assemble and solve the pde-010 N=4 two-dimensional Poisson example.

The script uses only the Python standard library. Its teaching focus is the
mapping from two-dimensional interior-node indices (i, j) to a one-dimensional
unknown vector and the assembly of the five-point finite-difference system.

The small dense system is solved by the same plain Gaussian elimination style
used in earlier articles. This is not a solver recommendation for large PDE
systems.
"""

import csv
from pathlib import Path


N = 4
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


def vector_index(i, j, n):
    """Map mathematical interior indices (i, j) to a zero-based Python index."""
    return (j - 1) * n + (i - 1)


def mathematical_index(i, j, n):
    """Return the one-based vector index used in the article."""
    return vector_index(i, j, n) + 1


def exact_solution(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def source_term(x, y):
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


def zero_boundary(x, y):
    return 0.0


def linear_exact_solution(x, y):
    return 1.0 + x + y


def zero_source(x, y):
    return 0.0


def assemble_dirichlet_system(N, source, boundary):
    """Assemble the dense five-point system for a square N-by-N partition."""
    h = 1.0 / N
    n = N - 1
    size = n * n
    A = [[0.0] * size for _ in range(size)]
    b = [0.0] * size
    rows = []

    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            row = vector_index(i, j, n)
            A[row][row] = 4.0
            b[row] = h * h * source(x, y)

            neighbor_ids = {
                "left_k": "",
                "right_k": "",
                "down_k": "",
                "up_k": "",
            }

            if i > 1:
                col = vector_index(i - 1, j, n)
                A[row][col] = -1.0
                neighbor_ids["left_k"] = mathematical_index(i - 1, j, n)
            else:
                b[row] += boundary(0.0, y)

            if i < N - 1:
                col = vector_index(i + 1, j, n)
                A[row][col] = -1.0
                neighbor_ids["right_k"] = mathematical_index(i + 1, j, n)
            else:
                b[row] += boundary(1.0, y)

            if j > 1:
                col = vector_index(i, j - 1, n)
                A[row][col] = -1.0
                neighbor_ids["down_k"] = mathematical_index(i, j - 1, n)
            else:
                b[row] += boundary(x, 0.0)

            if j < N - 1:
                col = vector_index(i, j + 1, n)
                A[row][col] = -1.0
                neighbor_ids["up_k"] = mathematical_index(i, j + 1, n)
            else:
                b[row] += boundary(x, 1.0)

            rows.append(
                {
                    "k": mathematical_index(i, j, n),
                    "i": i,
                    "j": j,
                    **neighbor_ids,
                    "rhs": b[row],
                }
            )

    return h, A, b, rows


def max_residual(A, x, b):
    """Return max_i |(A x - b)_i| for the assembled dense system."""
    maximum = 0.0
    for row in range(len(b)):
        value = 0.0
        for col in range(len(x)):
            value += A[row][col] * x[col]
        maximum = max(maximum, abs(value - b[row]))
    return maximum


def check_main_system(A, b, solution):
    n = N - 1

    expected_matrix = [
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
    assert A == expected_matrix

    expected_mapping = [
        ((1, 1), 1),
        ((2, 1), 2),
        ((3, 1), 3),
        ((1, 2), 4),
        ((2, 2), 5),
        ((3, 2), 6),
        ((1, 3), 7),
        ((2, 3), 8),
        ((3, 3), 9),
    ]
    for (i, j), k in expected_mapping:
        assert mathematical_index(i, j, n) == k

    # Consecutive vector positions 3 and 4 are not grid neighbors.
    assert A[2][3] == 0.0
    assert A[3][2] == 0.0

    # The center row k=5 couples only to k=2,4,5,6,8.
    center_row = A[4]
    nonzero_columns = []
    for col, value in enumerate(center_row):
        if value != 0.0:
            nonzero_columns.append(col + 1)
    assert nonzero_columns == [2, 4, 5, 6, 8]

    expected_solution = [
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
    for value, expected in zip(solution, expected_solution):
        assert abs(value - expected) < tol

    assert max_residual(A, solution, b) < tol


def check_nonzero_dirichlet_regression():
    """Check boundary-to-RHS assembly with u=1+x+y and -Delta u=0."""
    n = N - 1
    _, A, b, _ = assemble_dirichlet_system(N, zero_source, linear_exact_solution)
    solution = gaussian_elimination(A, b)

    for j in range(1, N):
        y = j / N
        for i in range(1, N):
            x = i / N
            k = vector_index(i, j, n)
            expected = linear_exact_solution(x, y)
            assert abs(solution[k] - expected) < tol

    assert max_residual(A, solution, b) < tol
    return solution


def write_results(h, A, b, solution, rows):
    article_root = Path(__file__).resolve().parents[1]
    results_dir = article_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    solution_output = results_dir / "n4_solution.csv"
    with solution_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "i", "j", "x", "y", "U", "exact", "error"])
        n = N - 1
        for j in range(1, N):
            y = j * h
            for i in range(1, N):
                x = i * h
                index = vector_index(i, j, n)
                value = solution[index]
                exact = exact_solution(x, y)
                writer.writerow(
                    [
                        index + 1,
                        i,
                        j,
                        f"{x:.12g}",
                        f"{y:.12g}",
                        f"{value:.12g}",
                        f"{exact:.12g}",
                        f"{value - exact:.12g}",
                    ]
                )

    rows_output = results_dir / "n4_stencil_rows.csv"
    with rows_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "k",
            "i",
            "j",
            "left_k",
            "right_k",
            "down_k",
            "up_k",
            "rhs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = row.copy()
            output_row["rhs"] = f"{row['rhs']:.12g}"
            writer.writerow(output_row)

    return solution_output, rows_output


def main():
    h, A, b, rows = assemble_dirichlet_system(N, source_term, zero_boundary)
    solution = gaussian_elimination(A, b)

    check_main_system(A, b, solution)
    regression_solution = check_nonzero_dirichlet_regression()
    solution_output, rows_output = write_results(h, A, b, solution, rows)

    max_error = 0.0
    n = N - 1
    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            index = vector_index(i, j, n)
            max_error = max(
                max_error,
                abs(solution[index] - exact_solution(x, y)),
            )

    regression_max_error = 0.0
    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            index = vector_index(i, j, n)
            regression_max_error = max(
                regression_max_error,
                abs(regression_solution[index] - linear_exact_solution(x, y)),
            )

    print(f"N={N}, interior_unknowns={(N - 1) ** 2}")
    print(f"max nodal error = {max_error:.3e}")
    print(f"max residual = {max_residual(A, solution, b):.3e}")
    print(f"nonzero-Dirichlet regression max error = {regression_max_error:.3e}")
    print(f"wrote {solution_output}")
    print(f"wrote {rows_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
