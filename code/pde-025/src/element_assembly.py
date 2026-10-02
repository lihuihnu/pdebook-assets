#!/usr/bin/env python3
"""Exact element integrals and small P1 solves for -u''=1 or x, u(0)=u(1)=0."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def validate_nodes(nodes):
    if len(nodes) < 3 or nodes[0] != 0.0 or nodes[-1] != 1.0:
        raise ValueError("need at least three nodes, with endpoints 0 and 1")
    for x in nodes:
        if not math.isfinite(x):
            raise ValueError("nodes must be finite")
    for i in range(len(nodes) - 1):
        if nodes[i + 1] <= nodes[i]:
            raise ValueError("nodes must be strictly increasing")


def solve_hand_example():
    a11, a12 = 8.0, -4.0
    a21, a22 = -4.0, 6.0
    b1, b2 = 1.0 / 4.0, 3.0 / 8.0
    m = a21 / a11
    u2 = (b2 - m * b1) / (a22 - m * a12)
    u1 = (b1 - a12 * u2) / a11
    return [0.0, u1, u2, 0.0]


def assemble_constant(nodes):
    validate_nodes(nodes)
    count = len(nodes)
    K = []
    for i in range(count):
        K.append([0.0] * count)
    load = [0.0] * count
    for e in range(count - 1):
        h = nodes[e + 1] - nodes[e]
        local = [[1.0 / h, -1.0 / h],
                 [-1.0 / h, 1.0 / h]]
        local_load = [h / 2.0, h / 2.0]
        indices = [e, e + 1]
        for alpha in range(2):
            i = indices[alpha]
            load[i] += local_load[alpha]
            for beta in range(2):
                j = indices[beta]
                K[i][j] += local[alpha][beta]
    return K, load


def element_load(a, b, source):
    h = b - a
    if source == "one":
        return [h / 2.0, h / 2.0]
    if source == "x":
        return [h * (2.0 * a + b) / 6.0, h * (a + 2.0 * b) / 6.0]
    raise ValueError("source must be one or x")


def assemble(nodes, source):
    K, load = assemble_constant(nodes)
    if source == "one":
        return K, load
    load = [0.0] * len(nodes)
    for e in range(len(nodes) - 1):
        local = element_load(nodes[e], nodes[e + 1], source)
        load[e] += local[0]
        load[e + 1] += local[1]
    return K, load


def solve_zero_boundary(K, load):
    # Keep the interior block; endpoint columns multiply prescribed zeros.
    count = len(load) - 2
    A = []
    b = []
    for i in range(1, count + 1):
        A.append(K[i][1:count + 1])
        b.append(load[i])
    # Gaussian elimination without pivoting: the interior matrix is SPD.
    for k in range(count):
        if not math.isfinite(A[k][k]) or A[k][k] <= 0.0:
            raise ArithmeticError("nonpositive or nonfinite elimination pivot")
        for i in range(k + 1, count):
            multiplier = A[i][k] / A[k][k]
            A[i][k] = 0.0
            for j in range(k + 1, count):
                A[i][j] -= multiplier * A[k][j]
            b[i] -= multiplier * b[k]
    U = [0.0] * count
    for i in range(count - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, count):
            total -= A[i][j] * U[j]
        U[i] = total / A[i][i]
    if not all(math.isfinite(value) for value in U):
        raise ArithmeticError("nonfinite solution")
    return [0.0] + U + [0.0]


def exact_value(x, source):
    if source == "one":
        return x * (1.0 - x) / 2.0
    if source == "x":
        return (x - x ** 3) / 6.0
    raise ValueError("unknown source")


def linear_value(nodes, U, x):
    for e in range(len(nodes) - 1):
        if x <= nodes[e + 1]:
            h = nodes[e + 1] - nodes[e]
            t = (x - nodes[e]) / h
            return (1.0 - t) * U[e] + t * U[e + 1]
    raise ValueError("point outside mesh")


def save_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(config, output):
    meshes = config["meshes"]
    sources = config["sources"]
    samples = config["profile_intervals"]
    if len({mesh["name"] for mesh in meshes}) != len(meshes):
        raise ValueError("duplicate mesh names")
    if sources != ["one", "x"] or not isinstance(samples, int) or samples < 2:
        raise ValueError("expected sources one,x and at least two profile intervals")
    for mesh in meshes:
        validate_nodes(mesh["nodes"])
    primary = next(mesh for mesh in meshes if mesh["name"] == "nonuniform")
    if primary["nodes"] != [0.0, 0.25, 0.5, 1.0]:
        raise ValueError("nonuniform is the fixed worked example")
    # Validate all inputs before creating or replacing result files.
    data = {key: [] for key in
            ("elements", "matrix", "loads", "nodes", "summary", "profiles", "assembly_trace")}
    for mesh in meshes:
        name, nodes = mesh["name"], mesh["nodes"]
        count = len(nodes)
        for source in sources:
            K, load = assemble(nodes, source)
            U = solve_zero_boundary(K, load)
            residual = []
            for i in range(count):
                value = -load[i]
                for j in range(count):
                    value += K[i][j] * U[j]
                    data["matrix"].append(dict(mesh=name, source=source, i=i, j=j,
                                               value=K[i][j]))
                residual.append(value)
                data["loads"].append(dict(mesh=name, source=source, i=i, value=load[i]))
                exact = exact_value(nodes[i], source)
                data["nodes"].append(dict(mesh=name, source=source, i=i, x=nodes[i],
                                          solution=U[i], exact=exact,
                                          error=U[i] - exact, residual=value,
                                          retained=int(0 < i < count - 1)))
            for e in range(count - 1):
                a, b = nodes[e], nodes[e + 1]
                h = b - a
                left, right = element_load(a, b, source)
                data["elements"].append(dict(mesh=name, source=source, element=e + 1,
                                             left=e, right=e + 1, a=a, b=b, h=h,
                                             kLL=1.0/h, kLR=-1.0/h,
                                             kRL=-1.0/h, kRR=1.0/h,
                                             bL=left, bR=right))
            energy = 0.0
            work = 0.0
            for i in range(count):
                work += load[i] * U[i]
                for j in range(count):
                    energy += U[i] * K[i][j] * U[j]
            slope_energy = 0.0
            for e in range(count - 1):
                slope_energy += (U[e + 1] - U[e]) ** 2 / (nodes[e + 1] - nodes[e])
            nodal_error = max(abs(U[i] - exact_value(nodes[i], source)) for i in range(count))
            interior_residual = max(abs(value) for value in residual[1:-1])
            data["summary"].append(dict(mesh=name, source=source, cells=count-1,
                                        nodal_error=nodal_error,
                                        interior_residual=interior_residual,
                                        left_residual=residual[0], right_residual=residual[-1],
                                        energy=energy, slope_energy=slope_energy, work=work,
                                        load_sum=sum(load)))
            if name == "nonuniform":
                if source == "one" and U != solve_hand_example():
                    raise ArithmeticError("fixed and assembled solutions disagree")
                for k in range(samples + 1):
                    x = k / samples
                    numerical = linear_value(nodes, U, x)
                    exact = exact_value(x, source)
                    data["profiles"].append(dict(source=source, x=x, solution=numerical,
                                                 exact=exact, error=exact-numerical))
    # A transparent trace records the six local updates for each primary element.
    K = [[0.0] * 4 for i in range(4)]
    load = [0.0] * 4
    nodes = primary["nodes"]
    for e in range(3):
        h = nodes[e + 1] - nodes[e]
        for alpha in range(2):
            i = e + alpha
            addition = h / 2.0
            before = load[i]
            load[i] += addition
            data["assembly_trace"].append(dict(element=e+1, kind="load", i=i, j="",
                                                before=before, addition=addition, after=load[i]))
            for beta in range(2):
                j = e + beta
                addition = (1.0 if alpha == beta else -1.0) / h
                before = K[i][j]
                K[i][j] += addition
                data["assembly_trace"].append(dict(element=e+1, kind="matrix", i=i, j=j,
                                                    before=before, addition=addition, after=K[i][j]))
    output.mkdir(parents=True, exist_ok=True)
    for key, rows in data.items():
        save_csv(output / (key + ".csv"), list(rows[0]), rows)
    lines = ["pde-025 element assembly", "Python " + sys.version.split()[0],
             "Equation: -u''=f on (0,1); u(0)=u(1)=0; f=1 or x.",
             "All element integrals are exact formulas; arithmetic is binary float.",
             "All mesh cases are assembled and solved; exact solutions only check the result.",
             "Primary fixed solution: " + repr(solve_hand_example())]
    for key, rows in data.items():
        lines.append(key + ".csv: " + str(len(rows)) + " rows")
    max_error = max(row["nodal_error"] for row in data["summary"])
    max_residual = max(row["interior_residual"] for row in data["summary"])
    lines.append(f"Maximum nodal error: {max_error:.3e}")
    lines.append(f"Maximum retained-row residual: {max_residual:.3e}")
    text = "\n".join(lines) + "\n"
    (output / "run.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config/meshes.json")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
