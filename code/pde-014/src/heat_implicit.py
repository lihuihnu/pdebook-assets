#!/usr/bin/env python3
"""Heat equation: transparent tridiagonal implicit stepping and reproducible data."""
import argparse
import csv
import json
import math
import platform
from pathlib import Path

def factor_tridiagonal(size, q):
    """分解主对角为 1+2q、相邻对角为 -q 的矩阵。"""
    pivots = [1.0 + 2.0 * q] * size
    multipliers = [0.0] * size
    for j in range(1, size):
        multipliers[j] = -q / pivots[j - 1]
        pivots[j] += q * multipliers[j]
    return pivots, multipliers


def solve_factored(rhs, q, pivots, multipliers):
    """复用消元系数；不改写传入的右端列表。"""
    size = len(rhs)
    reduced = rhs[:]
    for j in range(1, size):
        reduced[j] -= multipliers[j] * reduced[j - 1]
    solution = [0.0] * size
    solution[-1] = reduced[-1] / pivots[-1]
    for j in range(size - 2, -1, -1):
        solution[j] = (reduced[j] + q * solution[j + 1]) / pivots[j]
    return solution

def advance(values, ratio, theta, pivots, multipliers):
    """两个端点固定为零；theta=1 为隐式 Euler，0.5 为 CN。"""
    rhs = []
    for i in range(1, len(values) - 1):
        rhs.append(values[i] + (1.0 - theta) * ratio * (
            values[i - 1] - 2.0 * values[i] + values[i + 1]
        ))
    interior = solve_factored(rhs, theta * ratio, pivots, multipliers)
    return [0.0] + interior + [0.0]


def explicit_advance(values, ratio):
    next_values = [0.0] * len(values)
    for i in range(1, len(values) - 1):
        next_values[i] = values[i] + ratio * (
            values[i - 1] - 2.0 * values[i] + values[i + 1]
        )
    return next_values


def eigenvalue(mode, cells):
    return -4.0 * math.sin(mode * math.pi / (2.0 * cells)) ** 2


def amplification(mode, cells, ratio, theta):
    z = -ratio * eigenvalue(mode, cells)
    return (1.0 - (1.0 - theta) * z) / (1.0 + theta * z)


def reference_values(cells, alpha, time, modes, kind, step=0, ratio=0.0, theta=0.0):
    """Only references use modes; the numerical solver updates physical nodes."""
    values = [0.0] * (cells + 1)
    for mode, amplitude in modes:
        if kind == "pde":
            factor = math.exp(-alpha * (mode * math.pi) ** 2 * time)
        elif kind == "semidiscrete":
            factor = math.exp(alpha * cells ** 2 * eigenvalue(mode, cells) * time)
        elif kind == "fully_discrete":
            factor = amplification(mode, cells, ratio, theta) ** step
        else:
            raise ValueError("unknown reference kind")
        for i in range(1, cells):
            values[i] += amplitude * factor * math.sin(mode * math.pi * i / cells)
    return values


def max_difference(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def scaled_residual(old, new, ratio, theta):
    residual = 0.0
    scale = 1.0
    q = theta * ratio
    for i in range(1, len(old) - 1):
        rhs = old[i] + (1.0 - theta) * ratio * (old[i - 1] - 2.0 * old[i] + old[i + 1])
        lhs = (1.0 + 2.0 * q) * new[i] - q * new[i - 1] - q * new[i + 1]
        residual = max(residual, abs(lhs - rhs))
        scale = max(scale, abs(rhs) + (1.0 + 2.0 * q) * abs(new[i])
                    + q * (abs(new[i - 1]) + abs(new[i + 1])))
    return residual / scale


def self_checks():
    # Hand-derived two-unknown systems, including unchanged inputs.
    for theta, expected in ((1.0, 0.5), (0.5, 1.0 / 3.0)):
        old = [0.0, 1.0, 1.0, 0.0]
        pivots, multipliers = factor_tridiagonal(2, theta)
        new = advance(old, 1.0, theta, pivots, multipliers)
        assert max_difference(new, [0.0, expected, expected, 0.0]) < 2e-15
        assert old == [0.0, 1.0, 1.0, 0.0]
    # No diffusion, and the one-interior-node edge case.
    pivots, multipliers = factor_tridiagonal(3, 0.0)
    rhs = [1.0, -2.0, 3.0]
    assert solve_factored(rhs, 0.0, pivots, multipliers) == rhs
    assert rhs == [1.0, -2.0, 3.0]
    checks = 0
    for cells in (2, 3, 20):
        for theta in (0.5, 1.0):
            for ratio in (0.1, 4.0, 100.0):
                pivots, multipliers = factor_tridiagonal(cells - 1, theta * ratio)
                assert min(pivots) > 0.0
                for mode in range(1, cells):
                    old = reference_values(cells, 1.0, 0.0, [(mode, 1.0)], "pde")
                    new = advance(old, ratio, theta, pivots, multipliers)
                    factor = amplification(mode, cells, ratio, theta)
                    expected = [factor * value for value in old]
                    assert max_difference(new, expected) < 1e-12
                    assert scaled_residual(old, new, ratio, theta) < 1e-14
                    assert new[0] == new[-1] == 0.0
                    checks += 1
    return checks


def validate(config):
    cells = config["cells"]
    if type(cells) is not int or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    for key in ("alpha", "final_time"):
        if not isinstance(config[key], (float, int)) or isinstance(config[key], bool):
            raise ValueError(key + " must be a real number")
        if not math.isfinite(config[key]) or config[key] <= 0.0:
            raise ValueError(key + " must be finite and positive")
    if not config["damping_modes"]:
        raise ValueError("at least one damping mode is required")
    seen = set()
    for mode, amplitude in config["damping_modes"]:
        if type(mode) is not int or not 1 <= mode < cells or mode in seen:
            raise ValueError("modes must be distinct interior sine modes")
        if not math.isfinite(amplitude) or amplitude == 0.0:
            raise ValueError("each amplitude must be finite and nonzero")
        seen.add(mode)
    steps = config["convergence_steps"]
    if len(steps) < 2 or any(type(s) is not int or s < 1 for s in steps):
        raise ValueError("convergence_steps must contain positive integers")
    if any(steps[j] != 2 * steps[j - 1] for j in range(1, len(steps))):
        raise ValueError("convergence_steps must successively double")
    for key in ("implicit_steps", "explicit_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")


def simulate(case, config, method, steps, modes):
    cells, alpha, final_time = config["cells"], config["alpha"], config["final_time"]
    dt = final_time / steps
    ratio = alpha * dt * cells ** 2
    theta = {"FE": 0.0, "IE": 1.0, "CN": 0.5}[method]
    if method == "FE" and ratio > 0.5 + 1e-14:
        raise ValueError("the comparison FE run must satisfy r <= 1/2")
    if method != "FE":
        pivots, multipliers = factor_tridiagonal(cells - 1, theta * ratio)
    values = reference_values(cells, alpha, 0.0, modes, "pde")
    node_rows, history_rows, mode_rows = [], [], []
    largest_modal_defect = 0.0
    largest_residual = 0.0
    previous_energy = sum(v * v for v in values) / cells
    initial_max = max(values)
    initial_min = min(values)
    for n in range(steps + 1):
        time = n * dt
        pde = reference_values(cells, alpha, time, modes, "pde")
        semi = reference_values(cells, alpha, time, modes, "semidiscrete")
        discrete = reference_values(cells, alpha, time, modes, "fully_discrete", n, ratio, theta)
        defect = max_difference(values, discrete)
        largest_modal_defect = max(largest_modal_defect, defect)
        assert defect < 2e-11 * max(1.0, max(abs(v) for v in discrete))
        assert all(math.isfinite(v) for v in values)
        assert values[0] == values[-1] == 0.0
        energy = sum(v * v for v in values) / cells
        assert energy <= previous_energy + 2e-13
        previous_energy = energy
        if method in ("FE", "IE") and initial_min >= 0.0:
            assert min(values) >= -2e-13
            assert max(values) <= initial_max + 2e-13
        pde_error = max_difference(values, pde)
        time_error = max_difference(values, semi)
        history_rows.append([case, method, n, time, pde_error, time_error,
                             math.sqrt(energy), min(values), max(values)])
        for i in range(cells + 1):
            node_rows.append([case, method, n, time, i, i / cells, values[i],
                              pde[i], semi[i], discrete[i]])
        for mode, amplitude in modes:
            coefficient = 0.0
            for i in range(1, cells):
                coefficient += values[i] * math.sin(mode * math.pi * i / cells)
            coefficient *= 2.0 / cells
            closed = amplitude * amplification(mode, cells, ratio, theta) ** n
            assert abs(coefficient - closed) < 2e-11
            mode_rows.append([case, method, n, time, mode, coefficient,
                              coefficient / amplitude,
                              math.exp(-alpha * (mode * math.pi) ** 2 * time),
                              math.exp(alpha * cells ** 2 * eigenvalue(mode, cells) * time)])
        if n < steps:
            old = values
            if method == "FE":
                values = explicit_advance(old, ratio)
            else:
                values = advance(old, ratio, theta, pivots, multipliers)
            residual = scaled_residual(old, values, ratio, theta)
            largest_residual = max(largest_residual, residual)
            assert residual < 2e-14
    summary = [case, method, cells, steps, dt, ratio, final_time, pde_error, time_error,
               largest_modal_defect, largest_residual, 0 if method == "FE" else 1,
               0 if method == "FE" else steps]
    return node_rows, history_rows, mode_rows, summary


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=base / "config" / "implicit.json")
    parser.add_argument("--output", type=Path, default=base / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate(config)
    mode_checks = self_checks()
    nodes, histories, modes, summaries, convergence = [], [], [], [], []
    for method in ("FE", "IE", "CN"):
        steps = config["explicit_steps"] if method == "FE" else config["implicit_steps"]
        parts = simulate("damping_" + method, config, method, steps, config["damping_modes"])
        nodes.extend(parts[0])
        histories.extend(parts[1])
        modes.extend(parts[2])
        summaries.append(parts[3])
    for method in ("IE", "CN"):
        previous_error = None
        for steps in config["convergence_steps"]:
            parts = simulate("smooth_" + method + "_" + str(steps), config,
                             method, steps, [(1, 1.0)])
            nodes.extend(parts[0])
            histories.extend(parts[1])
            modes.extend(parts[2])
            summaries.append(parts[3])
            error = parts[3][8]
            error_ratio = "" if previous_error is None else previous_error / error
            convergence.append([method, steps, config["final_time"] / steps,
                                error, error_ratio, parts[3][7]])
            previous_error = error
    # Only after all numerical checks pass do we write the experiment data.
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "nodes.csv",
              ["case", "method", "step", "time", "node", "x", "value", "exact_pde",
               "exact_semidiscrete", "closed_fully_discrete"], nodes)
    write_csv(args.output / "history.csv",
              ["case", "method", "step", "time", "pde_max_error", "time_max_error",
               "discrete_l2_norm", "minimum", "maximum"], histories)
    write_csv(args.output / "modes.csv",
              ["case", "method", "step", "time", "mode", "coefficient", "normalized",
               "pde_normalized", "semidiscrete_normalized"], modes)
    write_csv(args.output / "summary.csv",
              ["case", "method", "cells", "steps", "dt", "r", "final_time", "pde_max_error",
               "time_max_error", "max_modal_defect", "max_scaled_residual",
               "factorizations", "linear_solves"], summaries)
    write_csv(args.output / "convergence.csv",
              ["method", "steps", "dt", "time_max_error", "halving_ratio", "pde_max_error"],
              convergence)
    report = [
        "pde-014 heat equation implicit stepping",
        "Python " + platform.python_version(),
        "Core dependencies: Python standard library only",
        "Configuration: " + json.dumps(config, ensure_ascii=False, sort_keys=True),
        "Self-checks: hand systems, unchanged inputs, zero diffusion, single interior node",
        "One-step mode/residual checks passed: " + str(mode_checks),
        "All trajectories: zero endpoints, finite values, nonincreasing discrete l2 norm",
        "All trajectories: node solver agrees with independent fully discrete modal formula",
        "FE/IE: nonnegative initial data stay nonnegative and below initial maximum",
        "Time errors compare to exact semidiscrete solution; PDE errors use continuous solution",
        "Rows: nodes=%d history=%d modes=%d summary=%d convergence=%d"
        % (len(nodes), len(histories), len(modes), len(summaries), len(convergence)),
    ]
    for row in summaries:
        report.append("%s: dt=%.8g r=%.8g E_pde=%.12e E_time=%.12e modal=%.3e residual=%.3e"
                      % (row[0], row[4], row[5], row[7], row[8], row[9], row[10]))
    report.append("PASS")
    text = "\n".join(report) + "\n"
    (args.output / "run.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
