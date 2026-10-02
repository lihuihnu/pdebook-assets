"""Transparent, unpreconditioned CSR/Jacobi/CG/GMRES teaching baseline."""
import argparse
import csv
import json
import math
import platform
from pathlib import Path


def four_cell_cg():
    def dot(a, b):
        total = 0.0
        for i in range(3):
            total += a[i] * b[i]
        return total

    def multiply(v):
        return [2 * v[0] - v[1],
                -v[0] + 2 * v[1] - v[2],
                -v[1] + 2 * v[2]]

    b = [1 / 16, 1 / 16, 1 / 16]
    u = [0.0, 0.0, 0.0]
    r = b.copy()
    p = r.copy()
    rr = dot(r, r)
    for k in range(3):
        ap = multiply(p)
        alpha = rr / dot(p, ap)
        for i in range(3):
            u[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
        au = multiply(u)
        true_r = [b[i] - au[i] for i in range(3)]
        if dot(true_r, true_r) <= 1e-24:
            return u
        new_rr = dot(r, r)
        beta = new_rr / rr
        for i in range(3):
            p[i] = r[i] + beta * p[i]
        rr = new_rr
    raise ArithmeticError("CG did not reach the tolerance")


def dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def norm(v):
    return math.sqrt(dot(v, v))


def csr_multiply(matrix, v):
    values, columns, row_ptr = matrix
    result = [0.0] * (len(row_ptr) - 1)
    for i in range(len(result)):
        for q in range(row_ptr[i], row_ptr[i + 1]):
            result[i] += values[q] * v[columns[q]]
    return result


def build_problem(N, problem):
    h = 1.0 / N
    c = problem["c"]
    values, columns, row_ptr = [], [], [0]
    b, exact = [], []
    for i in range(1, N):
        x = i * h
        if problem["source"] == "affine":
            f = 1 + x
            target = (4 * x - 3 * x * x - x ** 3) / 6
        else:
            f = 1 + c * (0.5 - x)
            target = x * (1 - x) / 2
        b.append(h * h * f)
        exact.append(target)
        # Only internal columns are stored. The zero boundaries add no load.
        for j, value in ((i - 2, -1 - c * h / 2),
                         (i - 1, 2.0), (i, -1 + c * h / 2)):
            if 0 <= j < N - 1 and value != 0:
                columns.append(j)
                values.append(value)
        row_ptr.append(len(values))
    return (values, columns, row_ptr), b, exact


def residual(matrix, b, u):
    au = csr_multiply(matrix, u)
    return [b[i] - au[i] for i in range(len(b))]


def record(history, snapshots, matrix, b, exact, u, k, cycle, inner,
           recursive_norm, matvecs, spd):
    r = residual(matrix, b, u)
    error = [u[i] - exact[i] for i in range(len(u))]
    energy = ""
    if spd:
        extended = [0.0] + error + [0.0]
        energy = math.sqrt(sum((extended[i + 1] - extended[i]) ** 2
                               for i in range(len(extended) - 1)))
    history.append((k, cycle, inner, norm(r), norm(r) / norm(b) if norm(b) else "",
                    recursive_norm, norm(error), energy, matvecs))
    # All Krylov iterates; selected Jacobi iterates avoid a bulky full trajectory.
    if cycle >= 0 or k <= 2 or (k and k & (k - 1) == 0):
        snapshots.append((k, u.copy()))
    return r


def finish(u, history, snapshots, status):
    if not snapshots or snapshots[-1][0] != history[-1][0]:
        snapshots.append((history[-1][0], u.copy()))
    return u, history, snapshots, status


def jacobi(matrix, b, exact, rtol, atol, maxiter):
    u = [0.0] * len(b)
    history, snapshots = [], []
    threshold = max(atol, rtol * norm(b))
    r = record(history, snapshots, matrix, b, exact, u, 0, -1, 0,
               norm(b), 1, True)
    if norm(r) <= threshold:
        return finish(u, history, snapshots, "converged")
    for k in range(1, maxiter + 1):
        # The diagonal of this Poisson matrix is two. All updates use old r.
        for i in range(len(u)):
            u[i] += r[i] / 2
        r = record(history, snapshots, matrix, b, exact, u, k, -1, k,
                   "", k + 1, True)
        if norm(r) <= threshold:
            return finish(u, history, snapshots, "converged")
    return finish(u, history, snapshots, "maxiter")


def cg(matrix, b, exact, rtol, atol, maxiter):
    u = [0.0] * len(b)
    history, snapshots = [], []
    threshold = max(atol, rtol * norm(b))
    r = record(history, snapshots, matrix, b, exact, u, 0, 0, 0,
               norm(b), 1, True)
    if norm(r) <= threshold:
        return finish(u, history, snapshots, "converged")
    p = r.copy()
    rr = dot(r, r)
    for k in range(1, maxiter + 1):
        ap = csr_multiply(matrix, p)
        denominator = dot(p, ap)
        if denominator <= 0 or not math.isfinite(denominator):
            return finish(u, history, snapshots, "nonpositive_curvature")
        alpha = rr / denominator
        for i in range(len(u)):
            u[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
        true_r = record(history, snapshots, matrix, b, exact, u, k, 0, k,
                        norm(r), 2 * k + 1, True)
        if norm(true_r) <= threshold:
            return finish(u, history, snapshots, "converged")
        new_rr = dot(r, r)
        if new_rr == 0 or not math.isfinite(new_rr):
            return finish(u, history, snapshots, "recursive_breakdown")
        beta = new_rr / rr
        for i in range(len(p)):
            p[i] = r[i] + beta * p[i]
        rr = new_rr
    return finish(u, history, snapshots, "maxiter")


def gmres(matrix, b, exact, rtol, atol, maxiter, restart, spd=False):
    """MGS Arnoldi with a second orthogonalization pass and Givens QR."""
    u = [0.0] * len(b)
    history, snapshots = [], []
    threshold = max(atol, rtol * norm(b))
    matvecs = 1
    r = record(history, snapshots, matrix, b, exact, u, 0, 0, 0,
               norm(b), matvecs, spd)
    if norm(r) <= threshold:
        return finish(u, history, snapshots, "converged")
    k, cycle = 0, 0
    while k < maxiter:
        cycle += 1
        base = u.copy()
        gamma = norm(r)
        basis = [[v / gamma for v in r]]
        size = min(restart, len(b), maxiter - k)
        H = [[0.0] * size for _ in range(size + 1)]
        cosines, sines = [], []
        g = [gamma] + [0.0] * size
        for j in range(size):
            w = csr_multiply(matrix, basis[j])
            matvecs += 1
            before = norm(w)
            # A second pass reduces loss of orthogonality in this small baseline.
            for pass_index in range(2):
                for i in range(j + 1):
                    projection = dot(basis[i], w)
                    H[i][j] += projection
                    for q in range(len(w)):
                        w[q] -= projection * basis[i][q]
            H[j + 1][j] = norm(w)
            breakdown = H[j + 1][j] <= 1e-14 * max(before, 1.0)
            if not breakdown:
                basis.append([v / H[j + 1][j] for v in w])
            for i in range(j):
                a, d = H[i][j], H[i + 1][j]
                H[i][j] = cosines[i] * a + sines[i] * d
                H[i + 1][j] = -sines[i] * a + cosines[i] * d
            radius = math.hypot(H[j][j], H[j + 1][j])
            if radius == 0 or not math.isfinite(radius):
                return finish(u, history, snapshots, "singular_small_problem")
            cosine, sine = H[j][j] / radius, H[j + 1][j] / radius
            cosines.append(cosine)
            sines.append(sine)
            H[j][j], H[j + 1][j] = radius, 0.0
            g[j + 1], g[j] = -sine * g[j], cosine * g[j]
            y = [0.0] * (j + 1)
            for i in range(j, -1, -1):
                value = g[i]
                for q in range(i + 1, j + 1):
                    value -= H[i][q] * y[q]
                y[i] = value / H[i][i]
            u = base.copy()
            for i in range(j + 1):
                for q in range(len(u)):
                    u[q] += y[i] * basis[i][q]
            k += 1
            matvecs += 1
            r = record(history, snapshots, matrix, b, exact, u, k, cycle, j + 1,
                       abs(g[j + 1]), matvecs, spd)
            if norm(r) <= threshold:
                return finish(u, history, snapshots, "converged")
            if breakdown:
                return finish(u, history, snapshots, "arnoldi_breakdown_unconverged")
    return finish(u, history, snapshots, "maxiter")


def validate(config):
    meshes = config["meshes"]
    if not meshes or meshes != sorted(set(meshes)):
        raise ValueError("meshes must be unique and increasing")
    if any(type(N) is not int or not 4 <= N <= 256 for N in meshes):
        raise ValueError("N must be an integer from 4 through 256")
    for key in ("rtol", "atol"):
        v = config[key]
        if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
            raise ValueError("tolerances must be finite and nonnegative")
    if config["rtol"] == config["atol"] == 0:
        raise ValueError("at least one tolerance must be positive")
    for key in ("jacobi_maxiter", "krylov_maxiter", "restart"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("iteration counts must be positive integers")
    expected = [
        {"name": "constant", "c": 0, "source": "constant"},
        {"name": "affine", "c": 0, "source": "affine"},
        {"name": "convection", "c": 8, "source": "quadratic_solution"}]
    if config["problems"] != expected:
        raise ValueError("this teaching experiment requires its three stated problems")


def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([format(v, ".17g") if type(v) is float else v for v in row])


def run(config, output):
    validate(config)
    matrix_rows, load_rows, histories, snapshots, nodes, summary = [], [], [], [], [], []
    fixed = four_cell_cg()
    if max(abs(a - b) for a, b in zip(fixed, [3 / 32, 1 / 8, 3 / 32])) > 1e-15:
        raise ArithmeticError("fixed four-cell example failed")
    for problem in config["problems"]:
        name, spd = problem["name"], problem["c"] == 0
        for N in config["meshes"]:
            matrix, b, exact = build_problem(N, problem)
            values, columns, ptr = matrix
            for i in range(N - 1):
                for q in range(ptr[i], ptr[i + 1]):
                    matrix_rows.append((name, N, i, columns[q], values[q], q))
                load_rows.append((name, N, i, (i + 1) / N, b[i], exact[i]))
            methods = ["jacobi", "cg", "gmres", "gmres_restart"] if spd else ["gmres", "gmres_restart"]
            for method in methods:
                args = (matrix, b, exact, config["rtol"], config["atol"])
                if method == "jacobi":
                    result = jacobi(*args, config["jacobi_maxiter"])
                elif method == "cg":
                    result = cg(*args, config["krylov_maxiter"])
                else:
                    restart = N - 1 if method == "gmres" else config["restart"]
                    result = gmres(*args, config["krylov_maxiter"], restart, spd)
                u, history, saved, status = result
                key = (name, N, method)
                histories.extend(key + row for row in history)
                for k, vector in saved:
                    snapshots.extend(key + (k, i, value) for i, value in enumerate(vector))
                for i, value in enumerate(u):
                    nodes.append(key + (i, value, exact[i], value - exact[i]))
                last = history[-1]
                summary.append(key + (N - 1, len(values), last[0], status, last[3],
                                      last[4], last[6], last[7], last[8]))
    # Exact two-dimensional stagnation example, stored including the failure status.
    rotation = ([ -1.0, 1.0], [1, 0], [0, 1, 2])
    for restart in (1, 2):
        key = ("rotation", 0, "gmres_" + str(restart))
        result = gmres(rotation, [1.0, 0.0], [0.0, -1.0],
                       config["rtol"], config["atol"], 6, restart)
        u, history, saved, status = result
        histories.extend(key + row for row in history)
        for k, vector in saved:
            snapshots.extend(key + (k, i, value) for i, value in enumerate(vector))
        for i, value in enumerate(u):
            nodes.append(key + (i, value, [0.0, -1.0][i], value - [0.0, -1.0][i]))
        last = history[-1]
        summary.append(key + (2, 2, last[0], status, last[3], last[4], last[6], "", last[8]))
    all_rows = [matrix_rows, load_rows, histories, snapshots, nodes, summary]
    if any(not math.isfinite(v) for group in all_rows for row in group
           for v in row if type(v) is float):
        raise ArithmeticError("nonfinite output")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matrix.csv", ["problem", "N", "i", "j", "value", "position"], matrix_rows)
    write_csv(output / "loads.csv", ["problem", "N", "i", "x", "b", "exact"], load_rows)
    write_csv(output / "history.csv",
              ["problem", "N", "method", "k", "cycle", "inner", "residual_norm", "relative_residual",
               "recursive_norm", "error_norm", "energy_error", "matvecs"], histories)
    write_csv(output / "iterates.csv", ["problem", "N", "method", "k", "i", "value"], snapshots)
    write_csv(output / "nodes.csv", ["problem", "N", "method", "i", "value", "exact", "error"], nodes)
    write_csv(output / "summary.csv",
              ["problem", "N", "method", "n", "nnz", "iterations", "status", "residual_norm",
               "relative_residual", "error_norm", "energy_error", "matvecs"], summary)
    counts = {name: len(rows) for name, rows in zip(
        ["matrix", "loads", "history", "iterates", "nodes", "summary"], all_rows)}
    log = ["Python " + platform.python_version(), "Core dependencies: standard library",
           "Row scaling: h^2 times differential equation; zero initial iterate",
           "Stopping: true Euclidean residual <= max(atol, rtol*norm(b))",
           "CG: recursive residual updates; true residual used for stopping",
           "GMRES: two-pass modified Gram-Schmidt, Givens rotations, true residual at each step",
           "Jacobi: Poisson only; GMRES(1) rotation intentionally reaches maxiter",
           "four_cell_cg: " + repr(fixed), "rows: " + json.dumps(counts, sort_keys=True)]
    for row in summary:
        log.append(f"{row[0]} N={row[1]} {row[2]} iterations={row[5]} status={row[6]}")
    (output / "run.txt").write_text("\n".join(log) + "\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    home = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=home / "config/problems.json")
    parser.add_argument("--output", type=Path, default=home / "results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
