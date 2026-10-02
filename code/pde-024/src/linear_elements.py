#!/usr/bin/env python3
"""Evaluate one-dimensional nodal hats and check a quadratic interpolant.

The general meshes use known exact nodal data, not a finite element system
solver. Only the one-interior-node example solves a scalar weak equation.
"""

import argparse
import csv
import json
import math
import platform
from pathlib import Path


def validate_nodes(nodes):
    if len(nodes) < 3 or any(not math.isfinite(x) for x in nodes):
        raise ValueError("need at least three finite nodes")
    if nodes[0] != 0.0 or nodes[-1] != 1.0:
        raise ValueError("this example uses endpoints 0 and 1")
    if any(b <= a for a, b in zip(nodes, nodes[1:])):
        raise ValueError("nodes must be strictly increasing")


def validate_values(nodes, values):
    validate_nodes(nodes)
    if len(values) != len(nodes) or any(not math.isfinite(v) for v in values):
        raise ValueError("one finite value is required for each node")
    if values[0] != 0.0 or values[-1] != 0.0:
        raise ValueError("this space has zero endpoint values")


def hat_value(nodes, j, x):
    """Evaluate an interior hat; nodes have already been validated."""
    if type(j) is not int or not 1 <= j < len(nodes) - 1:
        raise ValueError("j must be an interior node index")
    if not math.isfinite(x) or not nodes[0] <= x <= nodes[-1]:
        raise ValueError("x must lie in the mesh interval")
    if x <= nodes[j - 1] or x >= nodes[j + 1]:
        return 0.0
    if x <= nodes[j]:
        return (x - nodes[j - 1]) / (nodes[j] - nodes[j - 1])
    return (nodes[j + 1] - x) / (nodes[j + 1] - nodes[j])


def from_hats(nodes, values, x):
    """Use validated nodes and values, including zero endpoint values."""
    result = 0.0
    for j in range(1, len(nodes) - 1):
        result += values[j] * hat_value(nodes, j, x)
    return result


def local_value(nodes, values, x):
    """An independent evaluation route using one element's two endpoints."""
    if not math.isfinite(x) or not nodes[0] <= x <= nodes[-1]:
        raise ValueError("x must lie in the mesh interval")
    for i in range(1, len(nodes)):
        a, b = nodes[i - 1], nodes[i]
        if x <= b:
            t = (x - a) / (b - a)
            return (1.0 - t) * values[i - 1] + t * values[i]
    raise AssertionError("valid point was not assigned to an element")


def exact_value(x):
    return x * (1.0 - x) / 2.0


def element_errors(a, b, ua, ub):
    slope = (ub - ua) / (b - a)
    middle = (a + b) / 2.0
    midpoint_error = exact_value(middle) - (ua + ub) / 2.0
    # Integral of ((1/2 - x) - slope)^2 from a to b.
    slope_error_squared = ((b + slope - 0.5) ** 3
                           - (a + slope - 0.5) ** 3) / 3.0
    return middle, midpoint_error, slope_error_squared


def weak_pair(nodes, values, j):
    """Exact segment integrals for f=1 and test phi_j, without assembly."""
    left = 0.0
    right = 0.0
    for i in range(1, len(nodes)):
        a, b = nodes[i - 1], nodes[i]
        va = hat_value(nodes, j, a)
        vb = hat_value(nodes, j, b)
        slope = (values[i] - values[i - 1]) / (b - a)
        left += slope * (vb - va)
        right += (b - a) * (va + vb) / 2.0
    return left, right


def csv_output(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise ArithmeticError("non-finite output")
            writer.writerow({key: format(value, ".17g") if isinstance(value, float)
                             else value for key, value in row.items()})


def self_check():
    nodes = [0.0, 0.25, 0.5, 1.0]
    values = [0.0, 3.0 / 32, 1.0 / 8, 0.0]
    validate_values(nodes, values)
    assert hat_value(nodes, 2, 0.375) == hat_value(nodes, 2, 0.75) == 0.5
    assert from_hats(nodes, values, 0.375) == 7.0 / 64
    assert from_hats(nodes, values, 1.0) == 0.0
    for invalid in ([0.0, 0.5, 0.5, 1.0], [0.0, 0.5, float("nan"), 1.0]):
        try:
            validate_nodes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mesh accepted")


def run(config, output):
    nodes = config["primary_nodes"]
    refined = config["locally_refined_nodes"]
    levels = config["uniform_cells"]
    samples = config["profile_intervals"]
    validate_nodes(nodes)
    validate_nodes(refined)
    if len(nodes) != 4:
        raise ValueError("the primary figure uses exactly two interior hats")
    if (not isinstance(levels, list) or not levels
            or any(type(n) is not int or n < 2 for n in levels)
            or levels != sorted(set(levels))):
        raise ValueError("uniform cell counts must be distinct increasing integers >= 2")
    if type(samples) is not int or samples < 4:
        raise ValueError("profile_intervals must be an integer >= 4")
    if not config["hand_points"] or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in config["hand_points"]):
        raise ValueError("hand points must be finite and inside [0,1]")
    if not config["single_nodes"] or any(
            not math.isfinite(a) or not 0 < a < 1 for a in config["single_nodes"]):
        raise ValueError("single interior nodes must be finite and inside (0,1)")
    self_check()
    values = [exact_value(x) for x in nodes]
    validate_values(nodes, values)
    profiles = []
    node_checks = []
    for j in (1, 2):
        for i, x in enumerate(nodes):
            actual = hat_value(nodes, j, x)
            expected = float(i == j)
            node_checks.append(dict(j=j, i=i, x=x, value=actual,
                                    expected=expected, error=actual-expected))
    for k in range(samples + 1):
        x = k / samples
        phi1 = hat_value(nodes, 1, x)
        phi2 = hat_value(nodes, 2, x)
        value = from_hats(nodes, values, x)
        direct = local_value(nodes, values, x)
        profiles.append(dict(x=x, phi1=phi1, phi2=phi2,
                             term1=values[1]*phi1, term2=values[2]*phi2,
                             interpolant=value, local_value=direct,
                             exact=exact_value(x), error=exact_value(x)-value))
    hand = []
    for x in config["hand_points"]:
        value = from_hats(nodes, values, x)
        hand.append(dict(x=x, phi1=hat_value(nodes, 1, x),
                         phi2=hat_value(nodes, 2, x), interpolant=value,
                         exact=exact_value(x), error=exact_value(x)-value))
    meshes = [("nonuniform", nodes), ("locally_refined", refined)]
    for n in levels:
        meshes.append(("uniform_"+str(n), [i/n for i in range(n+1)]))
    elements = []
    summaries = []
    weak = []
    for name, grid in meshes:
        nodal = [exact_value(x) for x in grid]
        validate_values(grid, nodal)
        max_error = 0.0
        slope_squared = 0.0
        max_difference = 0.0
        nodal_error = max(abs(from_hats(grid, nodal, x)-u)
                          for x, u in zip(grid, nodal))
        for i in range(1, len(grid)):
            a, b = grid[i-1], grid[i]
            middle, error, slope_error = element_errors(a, b, nodal[i-1], nodal[i])
            max_error = max(max_error, abs(error))
            slope_squared += slope_error
            for t in (0.25, 0.5, 0.75):
                x = a + t*(b-a)
                max_difference = max(max_difference, abs(
                    from_hats(grid, nodal, x)-local_value(grid, nodal, x)))
            elements.append(dict(mesh=name, element=i, a=a, b=b, h=b-a,
                                 midpoint=middle, midpoint_error=error,
                                 slope_error_squared=slope_error))
        for j in range(1, len(grid)-1):
            left, right = weak_pair(grid, nodal, j)
            weak.append(dict(mesh=name, j=j, left=left, right=right,
                             residual=left-right))
        summaries.append(dict(mesh=name, cells=len(grid)-1,
                              h_max=max(b-a for a, b in zip(grid, grid[1:])),
                              nodal_error=nodal_error, max_error=max_error,
                              slope_error_squared=slope_squared,
                              slope_error=math.sqrt(slope_squared),
                              representation_difference=max_difference))
    single = []
    for a in config["single_nodes"]:
        derivative_square = 1.0/a + 1.0/(1.0-a)
        load = a/2.0 + (1.0-a)/2.0
        coefficient = load/derivative_square
        single.append(dict(a=a, derivative_square=derivative_square, load=load,
                           coefficient=coefficient, exact_node=exact_value(a),
                           difference=coefficient-exact_value(a)))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("basis_profiles", profiles), ("node_checks", node_checks),
                       ("hand_values", hand), ("element_errors", elements),
                       ("refinement", summaries), ("weak_checks", weak),
                       ("single_hat", single)):
        csv_output(output/(name+".csv"), data)
    log = (f"Python {platform.python_version()}; standard library only\n"
           "Problem: -u''=1, zero endpoints; exact u=x(1-x)/2\n"
           f"Meshes: {len(meshes)}; elements: {len(elements)}; weak pairs: {len(weak)}\n"
           f"Profiles: {len(profiles)}; cardinal checks: {len(node_checks)}; hand points: {len(hand)}\n"
           f"Single-interior-node scalar equations solved: {len(single)}\n"
           "General meshes use exact nodal data for interpolation; no matrix solve\n"
           "Max error evaluated at proven element maxima; slope error integrated analytically\n"
           "Self-check: nonuniform hats, reconstruction, endpoint and invalid meshes passed\n")
    (output/"run.txt").write_text(log, encoding="utf-8")
    print(log, end="")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root/"config/meshes.json")
    parser.add_argument("--output", type=Path, default=root/"results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
