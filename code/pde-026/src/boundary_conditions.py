#!/usr/bin/env python3
"""Small P1 solves for -u''=f on [0,1], with D/N boundary data.

Only Python's standard library is required. Neumann values are OUTWARD
normal derivatives: q_left=-u'(0), q_right=u'(1). No load projection is used.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def validate_nodes(nodes):
    if len(nodes) < 2 or nodes[0] != 0 or nodes[-1] != 1:
        raise ValueError("need endpoints 0 and 1 and at least one element")
    for x in nodes:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError("nodes must be finite numbers")
    for e in range(len(nodes) - 1):
        if nodes[e + 1] <= nodes[e]:
            raise ValueError("nodes must be strictly increasing")


def validate_case(case):
    if case["source"] not in ("zero", "one", "x"):
        raise ValueError("source must be zero, one or x")
    for side in ("left", "right"):
        boundary = case[side]
        value = boundary["value"]
        if boundary["type"] not in ("D", "N"):
            raise ValueError("boundary type must be D or N")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("boundary values must be finite numbers")
    pure_neumann = case["left"]["type"] == case["right"]["type"] == "N"
    representative = case.get("representative")
    if pure_neumann and representative not in ("pin_left", "zero_mean"):
        raise ValueError("pure Neumann needs pin_left or zero_mean representative")
    if not pure_neumann and representative is not None:
        raise ValueError("a representative is only used for pure Neumann")


def solve_fixed_dirichlet():
    g0, g1 = 1.0, 2.0
    right1 = 1.0 / 4.0 - (-4.0) * g0
    right2 = 3.0 / 8.0 - (-2.0) * g1
    m = -4.0 / 8.0
    u2 = (right2 - m * right1) / (6.0 - m * (-4.0))
    u1 = (right1 + 4.0 * u2) / 8.0
    return [g0, u1, u2, g1]


def element_load(a, b, source):
    h = b - a
    if source == "one":
        return [h / 2.0, h / 2.0]
    if source == "x":
        return [h * (2.0 * a + b) / 6.0, h * (a + 2.0 * b) / 6.0]
    if source == "zero":
        return [0.0, 0.0]
    raise ValueError("unknown source")


def assemble(nodes, source):
    validate_nodes(nodes)
    count = len(nodes)
    K = []
    for i in range(count):
        K.append([0.0] * count)
    load = [0.0] * count
    for e in range(count - 1):
        a, b = nodes[e], nodes[e + 1]
        local_load = element_load(a, b, source)
        indices = [e, e + 1]
        for alpha in range(2):
            i = indices[alpha]
            load[i] += local_load[alpha]
            for beta in range(2):
                j = indices[beta]
                sign = 1.0 if alpha == beta else -1.0
                K[i][j] += sign / (b - a)
    return K, load


def add_boundary_data(load, case):
    right = load[:]
    fixed, values = [], []
    for side, i in (("left", 0), ("right", len(load) - 1)):
        boundary = case[side]
        if boundary["type"] == "D":
            fixed.append(i)
            values.append(float(boundary["value"]))
        else:
            # Both signs are plus: the input already uses outward normals.
            right[i] += boundary["value"]
    return right, fixed, values


def reduce_dirichlet(K, load, fixed, values):
    free = []
    for i in range(len(load)):
        if i not in fixed:
            free.append(i)
    A = []
    right = []
    for i in free:
        row = []
        for j in free:
            row.append(K[i][j])
        value = load[i]
        for k in range(len(fixed)):
            j = fixed[k]
            value -= K[i][j] * values[k]
        A.append(row)
        right.append(value)
    return A, right, free


def solve_positive_system(A, right):
    # The reduced matrix is SPD. Work on copies so original rows remain testable.
    A = [row[:] for row in A]
    right = right[:]
    count = len(right)
    for k in range(count):
        if not math.isfinite(A[k][k]) or A[k][k] <= 0.0:
            raise ArithmeticError("nonpositive or nonfinite elimination pivot")
        for i in range(k + 1, count):
            multiplier = A[i][k] / A[k][k]
            A[i][k] = 0.0
            for j in range(k + 1, count):
                A[i][j] -= multiplier * A[k][j]
            right[i] -= multiplier * right[k]
    solution = [0.0] * count
    for i in range(count - 1, -1, -1):
        value = right[i]
        for j in range(i + 1, count):
            value -= A[i][j] * solution[j]
        solution[i] = value / A[i][i]
    if not all(math.isfinite(value) for value in solution):
        raise ArithmeticError("nonfinite solution")
    return solution


def restore_solution(count, fixed, values, free, solution):
    U = [0.0] * count
    for k in range(len(fixed)):
        U[fixed[k]] = values[k]
    for k in range(len(free)):
        U[free[k]] = solution[k]
    return U


def integral_mean(nodes, U):
    # Domain length is one. This is the integral of the P1 function, not mean(U).
    pieces = []
    for e in range(len(nodes) - 1):
        pieces.append((nodes[e + 1] - nodes[e]) * (U[e] + U[e + 1]) / 2.0)
    return math.fsum(pieces)


def original_residual(K, right, U):
    result = []
    for i in range(len(U)):
        products = [K[i][j] * U[j] for j in range(len(U))]
        result.append(math.fsum(products) - right[i])
    return result


def compatibility_tolerance(right):
    scale = max(1.0, math.fsum(abs(value) for value in right))
    return 64.0 * sys.float_info.epsilon * scale


def solve_problem(nodes, case):
    validate_case(case)
    K, body = assemble(nodes, case["source"])
    right, fixed, values = add_boundary_data(body, case)
    tolerance = compatibility_tolerance(right)
    defect = math.fsum(right)
    pure_neumann = not fixed
    if pure_neumann:
        if abs(defect) > tolerance:
            raise ValueError(f"incompatible pure Neumann data: load sum={defect:.17g}")
        # Select a representative only AFTER testing compatibility; never edit right.
        fixed, values = [0], [0.0]
    A, reduced_right, free = reduce_dirichlet(K, right, fixed, values)
    solution = solve_positive_system(A, reduced_right)
    U = restore_solution(len(nodes), fixed, values, free, solution)
    shift = 0.0
    if case.get("representative") == "zero_mean":
        shift = integral_mean(nodes, U)
        U = [value - shift for value in U]
    tested = list(range(len(nodes))) if pure_neumann else free
    residual = original_residual(K, right, U)
    return dict(K=K, body=body, right=right, A=A, reduced_right=reduced_right,
                free=free, U=U, tested=tested, residual=residual, shift=shift,
                defect=defect, tolerance=tolerance)


def exact_coefficients(case):
    # Derived from the differential equation; called only after the discrete solve.
    part = {"zero": [0.0, 0.0, 0.0, 0.0],
            "one": [0.0, 0.0, -0.5, 0.0],
            "x": [0.0, 0.0, 0.0, -1.0/6.0]}[case["source"]]
    left, right = case["left"], case["right"]
    if left["type"] == "D":
        part[0] = left["value"]
        if right["type"] == "D":
            part[1] = right["value"] - part[0] - part[2] - part[3]
        else:
            part[1] = right["value"] - 2.0*part[2] - 3.0*part[3]
    else:
        part[1] = -left["value"]
        if right["type"] == "D":
            part[0] = right["value"] - part[1] - part[2] - part[3]
        elif case["representative"] == "zero_mean":
            part[0] = -(part[1]/2.0 + part[2]/3.0 + part[3]/4.0)
    return part


def polynomial_value(coefficients, x):
    return sum(c * x**k for k, c in enumerate(coefficients))


def linear_value(nodes, U, x):
    if x < nodes[0] or x > nodes[-1]:
        raise ValueError("point outside mesh")
    for e in range(len(nodes) - 1):
        if x <= nodes[e + 1]:
            weight = (x - nodes[e]) / (nodes[e + 1] - nodes[e])
            return (1.0 - weight)*U[e] + weight*U[e + 1]
    raise ValueError("point outside mesh")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    meshes, cases = config["meshes"], config["cases"]
    if not meshes or not cases:
        raise ValueError("meshes and cases must not be empty")
    for group in (meshes, cases):
        if len({item["name"] for item in group}) != len(group):
            raise ValueError("duplicate mesh or case names")
    for mesh in meshes:
        validate_nodes(mesh["nodes"])
    for case in cases:
        validate_case(case)
    samples = config["profile_intervals"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("profile_intervals must be an integer at least two")
    profile_cases = config["profile_cases"]
    if not set(profile_cases) <= {case["name"] for case in cases}:
        raise ValueError("unknown profile case")
    primary = next(mesh for mesh in meshes if mesh["name"] == "nonuniform")
    if primary["nodes"] != [0, 0.25, 0.5, 1]:
        raise ValueError("nonuniform must be the worked example")
    data = {name: [] for name in ("matrix", "elements", "loads", "reduced", "nodes",
                                  "slopes", "summary", "profiles", "rejected", "diagnostics")}
    # Compute everything before writing, including all compatibility checks.
    for mesh in meshes:
        name, nodes = mesh["name"], mesh["nodes"]
        K, _ = assemble(nodes, "one")
        for i, row in enumerate(K):
            for j, value in enumerate(row):
                data["matrix"].append(dict(mesh=name, i=i, j=j, value=value))
        for source in sorted({case["source"] for case in cases}):
            for e in range(len(nodes)-1):
                a, b = nodes[e], nodes[e+1]
                left, right = element_load(a, b, source)
                data["elements"].append(dict(mesh=name, source=source, element=e+1,
                                             a=a, b=b, body_left=left, body_right=right))
        for case in cases:
            solved = solve_problem(nodes, case)
            U, residual = solved["U"], solved["residual"]
            coefficients = exact_coefficients(case)
            exact = [polynomial_value(coefficients, x) for x in nodes]
            shape_error = [(U[i]-U[0])-(exact[i]-exact[0]) for i in range(len(nodes))]
            for i, x in enumerate(nodes):
                data["loads"].append(dict(mesh=name, case=case["name"], i=i,
                                          body=solved["body"][i],
                                          boundary=solved["right"][i]-solved["body"][i],
                                          right=solved["right"][i]))
                data["nodes"].append(dict(mesh=name, case=case["name"], i=i, x=x,
                                          solution=U[i], exact=exact[i], error=U[i]-exact[i],
                                          shape_error=shape_error[i], residual=residual[i],
                                          tested=int(i in solved["tested"])))
            for k, i in enumerate(solved["free"]):
                for ell, j in enumerate(solved["free"]):
                    data["reduced"].append(dict(mesh=name, case=case["name"], i=i, j=j,
                                                value=solved["A"][k][ell],
                                                right=solved["reduced_right"][k]))
            for e in range(len(nodes)-1):
                h = nodes[e+1]-nodes[e]
                data["slopes"].append(dict(mesh=name, case=case["name"], element=e+1,
                                           h=h, slope=(U[e+1]-U[e])/h))
            data["summary"].append(dict(mesh=name, case=case["name"], elements=len(nodes)-1,
                                        source=case["source"], left_type=case["left"]["type"],
                                        left_value=case["left"]["value"],
                                        right_type=case["right"]["type"], right_value=case["right"]["value"],
                                        representative=case.get("representative", "none"),
                                        body_sum=math.fsum(solved["body"]),
                                        load_sum=solved["defect"], compatibility_tolerance=solved["tolerance"],
                                        removed_shift=solved["shift"], mean=integral_mean(nodes, U),
                                        nodal_error=max(abs(U[i]-exact[i]) for i in range(len(nodes))),
                                        shape_error=max(abs(value) for value in shape_error),
                                        tested_residual=max((abs(residual[i]) for i in solved["tested"]), default=0.0),
                                        h_last=nodes[-1]-nodes[-2],
                                        slope_last=(U[-1]-U[-2])/(nodes[-1]-nodes[-2])))
            if name == "nonuniform" and case["name"] in profile_cases:
                for k in range(samples+1):
                    x = k/samples
                    numerical = linear_value(nodes, U, x)
                    exact_at_x = polynomial_value(coefficients, x)
                    data["profiles"].append(dict(case=case["name"], x=x, solution=numerical,
                                                 exact=exact_at_x, error=exact_at_x-numerical))
            if name == "nonuniform" and case["name"] == "dd_one":
                if U != solve_fixed_dirichlet():
                    raise ArithmeticError("fixed and assembled Dirichlet solves disagree")
        # Deliberately incompatible data are refused by the normal solver.
        for source in ("one", "x"):
            bad = dict(name="incompatible_"+source, source=source,
                       left=dict(type="N", value=0), right=dict(type="N", value=0),
                       representative="pin_left")
            bad_K, body = assemble(nodes, source)
            try:
                solve_problem(nodes, bad)
            except ValueError as error:
                if not str(error).startswith("incompatible pure Neumann data"):
                    raise
                data["rejected"].append(dict(mesh=name, case=bad["name"],
                                             load_sum=math.fsum(body),
                                             tolerance=compatibility_tolerance(body), outcome="rejected"))
            else:
                raise ArithmeticError("incompatible Neumann problem was accepted")
            if name == "nonuniform":
                # An intentionally invalid diagnostic demonstrates the deleted-row defect.
                # It bypasses no public solver guard: the altered mixed system is built here.
                A, right, free = reduce_dirichlet(bad_K, body, [0], [0.0])
                altered = restore_solution(len(nodes), [0], [0.0], free,
                                           solve_positive_system(A, right))
                residual = original_residual(bad_K, body, altered)
                for i, x in enumerate(nodes):
                    data["diagnostics"].append(dict(kind="incompatible_pin_"+source, i=i, x=x,
                                                     solution=altered[i], residual=residual[i],
                                                     mean=integral_mean(nodes, altered)))
    # Two other deliberate errors in the worked example, saved as diagnostics only.
    nodes = primary["nodes"]
    K, body = assemble(nodes, "one")
    A, right, free = reduce_dirichlet(K, body, [0, 3], [0.0, 0.0])
    lost = restore_solution(4, [0, 3], [1.0, 2.0], free, solve_positive_system(A, right))
    residual = original_residual(K, body, lost)
    for i, x in enumerate(nodes):
        data["diagnostics"].append(dict(kind="lost_dirichlet_columns", i=i, x=x,
                                         solution=lost[i], residual=residual[i], mean=integral_mean(nodes, lost)))
    pinned_case = next(case for case in cases if case["name"] == "nn_pin_one")
    pinned = solve_problem(nodes, pinned_case)
    arithmetic = math.fsum(pinned["U"])/len(nodes)
    wrong_mean = [value-arithmetic for value in pinned["U"]]
    residual = original_residual(pinned["K"], pinned["right"], wrong_mean)
    for i, x in enumerate(nodes):
        data["diagnostics"].append(dict(kind="arithmetic_mean", i=i, x=x,
                                         solution=wrong_mean[i], residual=residual[i], mean=integral_mean(nodes, wrong_mean)))
    lines = ["pde-026 finite-element boundary conditions", "Python "+sys.version.split()[0],
             "Equation: -u''=f on (0,1), f=0,1,x; exact element integration, binary float.",
             "Neumann input: q_left=-u'(0), q_right=u'(1); both add to the boundary load.",
             "Pure Neumann: check load sum first; no source/load projection; check all original rows.",
             "Compatibility tolerance: 64*machine_epsilon*max(1,sum(abs(right))).",
             "Zero mean uses the integral of the continuous P1 function.",
             "Exact polynomials check computed solutions; they do not set solved nodal values.",
             "Diagnostic altered systems are not accepted solutions of incompatible Neumann problems.",
             "Fixed Dirichlet solution: "+repr(solve_fixed_dirichlet())]
    for key, rows in data.items():
        lines.append(key+".csv: "+str(len(rows))+" rows")
    lines.append("Solved cases: "+str(len(data["summary"])))
    lines.append(f'Maximum tested-row residual: {max(r["tested_residual"] for r in data["summary"]):.3e}')
    lines.append(f'Maximum error after aligning the left value: {max(r["shape_error"] for r in data["summary"]):.3e}')
    text = "\n".join(lines)+"\n"
    output.mkdir(parents=True, exist_ok=True)
    for key, rows in data.items():
        save_csv(output/(key+".csv"), rows)
    (output/"run.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root/"config/problems.json")
    parser.add_argument("--output", type=Path, default=root/"results")
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text(encoding="utf-8")), args.output)
    except (ValueError, ArithmeticError, KeyError, StopIteration) as error:
        parser.exit(1, "error: "+str(error)+"\n")


if __name__ == "__main__":
    main()
