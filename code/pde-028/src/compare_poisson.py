"""Three specified nodal discretizations of -u''=f, u(0)=u(1)=0.

Teaching core: standard library only. All integrals here have polynomial
integrands; five-point Gauss integration is exact through degree nine in
exact arithmetic. Plotting is a separate entry point.
"""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import platform


def four_cell_values(method):
    b = [1 / 256, 4 / 256, 9 / 256]
    if method == "fv":
        correction = 1 / 3072
    elif method == "fe":
        correction = 1 / 1536
    elif method == "fd":
        correction = 0.0
    else:
        raise ValueError("method must be fd, fv or fe")
    for i in range(3):
        b[i] += correction
    u2 = (b[0] + 2 * b[1] + b[2]) / 2
    u1 = (b[0] + u2) / 2
    u3 = (b[2] + u2) / 2
    return [0.0, u1, u2, u3, 0.0]


def polynomial(coefficients, x):
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def gauss5(function, left, right):
    inner = math.sqrt(5 - 2 * math.sqrt(10 / 7)) / 3
    outer = math.sqrt(5 + 2 * math.sqrt(10 / 7)) / 3
    inner_weight = (322 + 13 * math.sqrt(70)) / 900
    outer_weight = (322 - 13 * math.sqrt(70)) / 900
    points = [-outer, -inner, 0.0, inner, outer]
    weights = [outer_weight, inner_weight, 128 / 225,
               inner_weight, outer_weight]
    center = (left + right) / 2
    radius = (right - left) / 2
    value = 0.0
    for point, weight in zip(points, weights):
        value += weight * function(center + radius * point)
    return radius * value


def exact_coefficients(source):
    coefficients = [0.0] * (len(source) + 2)
    for degree, coefficient in enumerate(source):
        coefficients[degree + 2] = -coefficient / ((degree + 1) * (degree + 2))
        coefficients[1] -= coefficients[degree + 2]
    return coefficients


def row_coefficients(nodes, i):
    left = nodes[i] - nodes[i - 1]
    right = nodes[i + 1] - nodes[i]
    return -1 / left, 1 / left + 1 / right, -1 / right


def source_loads(nodes, coefficients, i):
    x = nodes[i]
    left, right = nodes[i - 1], nodes[i + 1]
    cv_left, cv_right = (left + x) / 2, (x + right) / 2
    fd = (cv_right - cv_left) * polynomial(coefficients, x)
    fv = gauss5(lambda y: polynomial(coefficients, y), cv_left, cv_right)
    fe_left = gauss5(lambda y: polynomial(coefficients, y) * (y - left) / (x - left),
                     left, x)
    fe_right = gauss5(lambda y: polynomial(coefficients, y) * (right - y) / (right - x),
                      x, right)
    return fd, fv, fe_left + fe_right


def solve_nodes(nodes, loads):
    """Forward elimination and back substitution for the internal system."""
    count = len(nodes) - 2
    diagonal = []
    upper = []
    rhs = list(loads)
    for i in range(1, len(nodes) - 1):
        lower_i, diagonal_i, upper_i = row_coefficients(nodes, i)
        if i > 1:
            factor = lower_i / diagonal[i - 2]
            diagonal_i -= factor * upper[i - 2]
            rhs[i - 1] -= factor * rhs[i - 2]
        if not math.isfinite(diagonal_i) or diagonal_i <= 0:
            raise ArithmeticError("nonpositive elimination pivot")
        diagonal.append(diagonal_i)
        upper.append(upper_i)
    values = [0.0] * (count + 2)
    for j in range(count - 1, -1, -1):
        values[j + 1] = (rhs[j] - upper[j] * values[j + 2]) / diagonal[j]
    if not all(math.isfinite(value) for value in values):
        raise ArithmeticError("nonfinite solution")
    return values


def linear_value(nodes, values, x):
    for e in range(len(nodes) - 1):
        if x <= nodes[e + 1]:
            fraction = (x - nodes[e]) / (nodes[e + 1] - nodes[e])
            return values[e] * (1 - fraction) + values[e + 1] * fraction
    return values[-1]


def l2_error(nodes, values, exact):
    squared = 0.0
    for e in range(len(nodes) - 1):
        left, right = nodes[e], nodes[e + 1]
        def squared_error(x):
            weight = (x - left) / (right - left)
            approximate = (1 - weight) * values[e] + weight * values[e + 1]
            return (approximate - polynomial(exact, x)) ** 2
        squared += gauss5(squared_error, left, right)
    return math.sqrt(squared)


def read_config(path):
    config = json.loads(path.read_text())
    if config.get("methods") != ["fd", "fv", "fe"]:
        raise ValueError("methods must be fd, fv, fe in this order")
    if not isinstance(config.get("profile_points"), int) or config["profile_points"] < 3:
        raise ValueError("profile_points must be an integer at least three")
    meshes = []
    names = set()
    for mesh in config["meshes"]:
        name = mesh["name"]
        if name in names or not isinstance(name, str) or not name:
            raise ValueError("mesh names must be unique nonempty strings")
        names.add(name)
        nodes = [float(Fraction(str(value))) for value in mesh["nodes"]]
        if len(nodes) < 3 or len(nodes) > 257:
            raise ValueError("mesh must have between three and 257 nodes")
        if nodes[0] != 0 or nodes[-1] != 1:
            raise ValueError("mesh endpoints must be zero and one")
        if not all(math.isfinite(x) for x in nodes):
            raise ValueError("mesh nodes must be finite")
        if not all(nodes[i] < nodes[i + 1] for i in range(len(nodes) - 1)):
            raise ValueError("mesh nodes must be strictly increasing")
        if mesh["family"] not in ("uniform", "graded", "alternating"):
            raise ValueError("unknown mesh family")
        if mesh["family"] == "uniform":
            step = 1 / (len(nodes) - 1)
            if not all(abs(x - i * step) < 1e-13 for i, x in enumerate(nodes)):
                raise ValueError("uniform family requires equally spaced nodes")
        meshes.append((name, mesh["family"], nodes))
    if not meshes:
        raise ValueError("at least one mesh is required")
    sources = []
    source_names = set()
    for source in config["sources"]:
        name = source["name"]
        if name in source_names or not isinstance(name, str) or not name:
            raise ValueError("source names must be unique nonempty strings")
        source_names.add(name)
        coefficients = [float(value) for value in source["coefficients"]]
        if not 1 <= len(coefficients) <= 3:
            raise ValueError("sources must be polynomials of degree at most two")
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("source coefficients must be finite")
        sources.append((name, coefficients))
    if not sources:
        raise ValueError("at least one source is required")
    if not all(name in names for name in config["profile_meshes"]):
        raise ValueError("profile mesh is missing")
    return config, meshes, sources


def write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path, output):
    config, meshes, sources = read_config(config_path)
    for degree in range(10):
        value = gauss5(lambda x: x ** degree, -1, 1)
        expected = 2 / (degree + 1) if degree % 2 == 0 else 0.0
        if abs(value - expected) > 2e-14:
            raise ArithmeticError("Gauss5 polynomial moment failed")
    tables = {name: [] for name in ("matrix", "loads", "nodes", "fluxes",
                                    "balances", "boundary_balance", "profiles", "summary")}
    previous_errors = {}
    max_residual = 0.0
    for mesh_name, family, nodes in meshes:
        count = len(nodes) - 1
        hmax = max(nodes[i + 1] - nodes[i] for i in range(count))
        for i in range(1, count):
            coefficients = row_coefficients(nodes, i)
            for j, value in zip((i - 1, i, i + 1), coefficients):
                tables["matrix"].append(dict(mesh=mesh_name, i=i, j=j, value=value))
        for source_name, source in sources:
            exact = exact_coefficients(source)
            load_columns = [[], [], []]
            for i in range(1, count):
                fd, fv, fe = source_loads(nodes, source, i)
                for column, value in zip(load_columns, (fd, fv, fe)):
                    column.append(value)
                tables["loads"].append(dict(mesh=mesh_name, source=source_name,
                    i=i, x=nodes[i], left=(nodes[i - 1] + nodes[i]) / 2,
                    right=(nodes[i] + nodes[i + 1]) / 2, fd=fd, fv=fv, fe=fe))
            for method, loads in zip(config["methods"], load_columns):
                values = solve_nodes(nodes, loads)
                if nodes == [0.0, 0.25, 0.5, 0.75, 1.0] and source == [0.0, 0.0, 1.0]:
                    fixed = four_cell_values(method)
                    if max(abs(a - b) for a, b in zip(values, fixed)) > 1e-13:
                        raise ArithmeticError("fixed four-cell solution disagrees")
                q = []
                for e in range(count):
                    midpoint = (nodes[e] + nodes[e + 1]) / 2
                    numerical = -(values[e + 1] - values[e]) / (nodes[e + 1] - nodes[e])
                    exact_q = 0.0
                    for degree in range(1, len(exact)):
                        exact_q -= degree * exact[degree] * midpoint ** (degree - 1)
                    q.append(numerical)
                    tables["fluxes"].append(dict(mesh=mesh_name, source=source_name,
                        method=method, e=e, x=midpoint, q=numerical, exact_q=exact_q))
                residuals, cv_residuals = [], []
                for i in range(1, count):
                    difference = q[i] - q[i - 1]
                    cv_source = load_columns[1][i - 1]
                    algebraic = difference - loads[i - 1]
                    cv_residual = difference - cv_source
                    residuals.append(abs(algebraic))
                    cv_residuals.append(abs(cv_residual))
                    tables["balances"].append(dict(mesh=mesh_name, source=source_name,
                        method=method, i=i, x=nodes[i], q_left=q[i - 1], q_right=q[i],
                        difference=difference, cv_source=cv_source, method_load=loads[i - 1],
                        algebraic_residual=algebraic, cv_residual=cv_residual))
                left_source = gauss5(lambda x: polynomial(source, x), 0, (nodes[0] + nodes[1]) / 2)
                right_source = gauss5(lambda x: polynomial(source, x), (nodes[-2] + nodes[-1]) / 2, 1)
                q_left_boundary = q[0] - left_source
                q_right_boundary = q[-1] + right_source
                total_source = gauss5(lambda x: polynomial(source, x), 0, 1)
                global_residual = q_right_boundary - q_left_boundary - total_source
                tables["boundary_balance"].append(dict(mesh=mesh_name, source=source_name,
                    method=method, left_half_source=left_source, right_half_source=right_source,
                    q_left_boundary=q_left_boundary, q_right_boundary=q_right_boundary,
                    total_source=total_source, global_residual=global_residual))
                errors = []
                for i, x in enumerate(nodes):
                    reference = polynomial(exact, x)
                    error = values[i] - reference
                    errors.append(abs(error))
                    tables["nodes"].append(dict(mesh=mesh_name, source=source_name,
                        method=method, i=i, x=x, value=values[i], exact=reference, error=error))
                l2 = l2_error(nodes, values, exact)
                residual = max(residuals)
                max_residual = max(max_residual, residual)
                order = ""
                if family == "uniform":
                    key = source_name, method
                    if key in previous_errors:
                        old_h, old_l2 = previous_errors[key]
                        if hmax >= old_h:
                            raise ValueError("uniform meshes must be ordered coarse to fine")
                        order = math.log(old_l2 / l2) / math.log(old_h / hmax)
                    previous_errors[key] = hmax, l2
                tables["summary"].append(dict(mesh=mesh_name, family=family, source=source_name,
                    method=method, N=count, hmax=hmax, node_error=max(errors), l2_error=l2,
                    algebraic_residual=residual, cv_residual=max(cv_residuals),
                    global_residual=global_residual, l2_order=order))
                if mesh_name in config["profile_meshes"]:
                    for j in range(config["profile_points"]):
                        x = j / (config["profile_points"] - 1)
                        approximate = linear_value(nodes, values, x)
                        reference = polynomial(exact, x)
                        tables["profiles"].append(dict(mesh=mesh_name, source=source_name,
                            method=method, x=x, value=approximate, exact=reference,
                            error=approximate - reference))
    if max_residual > 1e-10:
        raise ArithmeticError("linear solve residual exceeds tolerance")
    # Validate and calculate before creating any output directory.
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), list(rows[0]), rows)
    lines = [f"Python {platform.python_version()}", "Model: -u''=f on (0,1), u(0)=u(1)=0",
             "FV unknowns: nodal values on midpoint dual control volumes",
             "FE: continuous P1 with accurate polynomial loads",
             "Function comparison: same piecewise linear reconstruction for all methods",
             "Gauss5: exact polynomial integration through degree nine, up to roundoff",
             "Gauss5 moments: ten monomials of degree zero through nine passed",
             f"Meshes: {len(meshes)}; sources: {len(sources)}; methods: 3",
             f"Solves: {len(tables['summary'])}",
             f"Maximum algebraic residual: {max_residual:.17g}"]
    for name, rows in tables.items():
        lines.append(f"{name}.csv: {len(rows)} rows")
    (output / "run.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here / "config/problems.json")
    parser.add_argument("--output", type=Path, default=here / "results")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
