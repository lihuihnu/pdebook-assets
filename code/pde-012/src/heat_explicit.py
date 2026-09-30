"""Explicit Euler / centered-space heat equation; standard library only."""
import csv
import json
import math
import platform
from pathlib import Path


def advance(values, ratio):
    """One step with zero Dirichlet boundaries; preserve the old layer."""
    next_values = [0.0] * len(values)
    for i in range(1, len(values) - 1):
        next_values[i] = values[i] + ratio * (
            values[i - 1] - 2.0 * values[i] + values[i + 1]
        )
    return next_values


def solve(intervals, steps, final_time, alpha):
    if not isinstance(intervals, int) or intervals < 2:
        raise ValueError("intervals must be an integer >= 2")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    h = 1.0 / intervals
    dt = final_time / steps
    ratio = alpha * dt / h**2
    if not math.isfinite(ratio) or ratio > 0.5:
        raise ValueError("this experiment requires alpha * dt / h**2 <= 0.5")
    values = [0.0] * (intervals + 1)
    for i in range(1, intervals):
        values[i] = math.sin(math.pi * (i * h))
    history = [values.copy()]
    for n in range(steps):
        values = advance(values, ratio)
        if not all(math.isfinite(value) for value in values):
            raise ArithmeticError("non-finite temperature")
        history.append(values)
    return h, dt, ratio, history


def check_implementation():
    old = [0.0, 1.0, 2.0, 1.0, 0.0]
    assert advance(old, 0.25) == [0.0, 1.0, 1.5, 1.0, 0.0]
    assert old == [0.0, 1.0, 2.0, 1.0, 0.0]
    h, dt, ratio, history = solve(20, 100, 0.1, 1.0)
    factor = 1.0 - 4.0 * ratio * math.sin(math.pi * h / 2.0)**2
    for n, values in enumerate(history):
        assert values[0] == values[-1] == 0.0
        assert min(values) >= 0.0
        if n:
            assert max(values) <= max(history[n - 1]) + 1e-14
        for i, value in enumerate(values):
            expected = factor**n * math.sin(math.pi * (i * h))
            assert abs(value - expected) < 2e-14
    try:
        solve(20, 10, 0.1, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe demonstration parameters were accepted")


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "heat.json").read_text())
    check_implementation()
    h, dt, ratio, history = solve(
        config["intervals"], config["steps"], config["final_time"], config["alpha"]
    )
    output = root / "results"
    output.mkdir(exist_ok=True)
    summaries = []
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "t", "i", "x", "numerical", "exact", "error"])
        for n, values in enumerate(history):
            t = n * dt
            error_inf = 0.0
            for i, value in enumerate(values):
                x = i * h
                exact = math.exp(-config["alpha"] * math.pi**2 * t) * math.sin(math.pi * x)
                if i == 0 or i == config["intervals"]:
                    exact = 0.0
                error = value - exact
                error_inf = max(error_inf, abs(error))
                writer.writerow([n, format(t, ".17g"), i, format(x, ".17g"),
                                 format(value, ".17g"), format(exact, ".17g"),
                                 format(error, ".17g")])
            if n in config["snapshot_steps"]:
                summaries.append([n, t, values[config["intervals"] // 2],
                                  math.exp(-config["alpha"] * math.pi**2 * t),
                                  error_inf])
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "t", "center_numerical", "center_exact", "error_inf"])
        writer.writerows(summaries)
    lines = [f"Python {platform.python_version()}",
             f"N={config['intervals']}, M={config['steps']}, h={h:g}, dt={dt:g}, r={ratio:.12g}",
             "checks: hand step, unchanged old layer, boundaries, positivity, maximum, discrete sine mode, parameter guard PASS"]
    for n, t, numerical, exact, error in summaries:
        lines.append(f"n={n:3d} t={t:.3f} center={numerical:.9f} exact={exact:.9f} error_inf={error:.9e}")
    text = "\n".join(lines) + "\n"
    (output / "run.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

