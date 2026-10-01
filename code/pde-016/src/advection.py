#!/usr/bin/env python3
"""Periodic constant-speed advection: transparent standard-library experiments.

Run from any directory. The default configuration and outputs are relative to
this file. Plotting is separate; neither NumPy nor Matplotlib is needed here.
"""

import argparse
import cmath
import csv
import json
import math
import platform
from pathlib import Path


def upwind_step(current, courant):
    """周期一阶迎风；稳定计算要求 abs(courant) <= 1。"""
    count = len(current)
    following = [0.0] * count
    if courant >= 0.0:
        for i in range(count):
            upstream = (i - 1) % count
            following[i] = (1.0 - courant) * current[i] + courant * current[upstream]
    else:
        weight = -courant
        for i in range(count):
            upstream = (i + 1) % count
            following[i] = (1.0 - weight) * current[i] + weight * current[upstream]
    return following


def centered_step(current, courant):
    """FTCS comparison; deliberately permits the unstable configuration."""
    count = len(current)
    following = [0.0] * count
    for i in range(count):
        left = (i - 1) % count
        right = (i + 1) % count
        following[i] = current[i] - 0.5 * courant * (current[right] - current[left])
    return following


def pulse(x, center, half_width, power):
    """A compact pulse, extended with period one; support stays within (0,1)."""
    distance = (x % 1.0 - center) / half_width
    if abs(distance) >= 1.0:
        return 0.0
    return (1.0 - distance * distance) ** power


def check_grid(count, final_time, steps, speed):
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError("N must be an integer >= 4")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("T must be finite and positive")
    if not math.isfinite(speed):
        raise ValueError("speed must be finite")
    h = 1.0 / count
    dt = final_time / steps
    courant = speed * dt / h
    if not math.isfinite(courant):
        raise ValueError("non-finite Courant number")
    return h, dt, courant


def l2(values, h):
    return math.sqrt(h * math.fsum(value * value for value in values))


def simulate(case, scheme, count, final_time, steps, speed, initial, exact,
             amplification=None):
    h, dt, courant = check_grid(count, final_time, steps, speed)
    if scheme not in ("upwind", "centered"):
        raise ValueError("unknown scheme")
    if scheme == "upwind" and abs(courant) > 1.0:
        raise ValueError("production upwind run requires abs(courant) <= 1")
    if len(initial) != count or not all(math.isfinite(v) for v in initial):
        raise ValueError("initial data must contain N finite values")
    current = list(initial)
    initial_norm = l2(current, h)
    initial_total = h * math.fsum(current)
    initial_min, initial_max = min(current), max(current)
    total_drift = 0.0
    max_bound_violation = 0.0
    max_prediction_relative_error = 0.0
    history = []
    advance = upwind_step if scheme == "upwind" else centered_step
    for step in range(steps + 1):
        time = step * dt
        if not all(math.isfinite(value) for value in current):
            raise ArithmeticError(f"non-finite result: {case}, step {step}")
        norm = l2(current, h)
        total = h * math.fsum(current)
        error = max(abs(current[i] - exact(i * h, time)) for i in range(count))
        if not all(math.isfinite(value) for value in (norm, total, error)):
            raise ArithmeticError(f"non-finite diagnostic: {case}, step {step}")
        total_drift = max(total_drift, abs(total - initial_total))
        violation = max(0.0, max(current) - initial_max, initial_min - min(current))
        max_bound_violation = max(max_bound_violation, violation)
        norm_ratio = norm / initial_norm if initial_norm else 0.0
        predicted = amplification ** step if amplification is not None else ""
        if amplification is not None:
            max_prediction_relative_error = max(
                max_prediction_relative_error, abs(norm_ratio / predicted - 1.0))
        history.append({"case": case, "scheme": scheme, "step": step, "t": time,
                        "minimum": min(current), "maximum": max(current),
                        "total": total, "l2": norm, "norm_ratio": norm_ratio,
                        "predicted_ratio": predicted, "max_error": error})
        if step < steps:
            current = advance(current, courant)
    nodes = [{"case": case, "scheme": scheme, "N": count, "i": i, "x": i * h,
              "initial": initial[i], "numerical": current[i],
              "exact": exact(i * h, final_time)} for i in range(count)]
    summary = {"case": case, "scheme": scheme, "N": count, "h": h, "dt": dt,
               "steps": steps, "T": final_time, "c": speed, "courant": courant,
               "max_error": history[-1]["max_error"], "minimum": min(current),
               "maximum": max(current), "initial_total": initial_total,
               "max_total_drift": total_drift,
               "max_bound_violation": max_bound_violation,
               "final_norm_ratio": history[-1]["norm_ratio"],
               "max_prediction_relative_error": max_prediction_relative_error
               if amplification is not None else ""}
    if scheme == "upwind" and max_bound_violation > 5e-14 * max(1.0, initial_norm):
        raise AssertionError("upwind violated initial extrema")
    if amplification is not None and max_prediction_relative_error > 2e-12:
        raise AssertionError("node update disagrees with modal prediction")
    return nodes, history, summary


def assert_close(actual, expected, tolerance=2e-13):
    if len(actual) != len(expected):
        raise AssertionError("length mismatch")
    error = max(abs(a - b) for a, b in zip(actual, expected))
    if error > tolerance:
        raise AssertionError(f"maximum error {error} exceeds {tolerance}")


def self_check():
    """Independent small problems and closed forms; not a second time loop."""
    original = [0.0, 1.0, 0.0, 0.0]
    assert_close(upwind_step(original, 0.5), [0.0, 0.5, 0.5, 0.0])
    assert_close(upwind_step(original, -0.5), [0.5, 0.5, 0.0, 0.0])
    assert_close(centered_step(original, 0.5), [-0.25, 1.0, 0.25, 0.0])
    assert original == [0.0, 1.0, 0.0, 0.0]
    data = [0.2, -0.6, 1.2, 0.3, 0.0, -0.1, 0.8]
    for nu in (0.0, 1.0, -1.0):
        shift = int(nu)
        assert_close(upwind_step(data, nu), [data[(i - shift) % 7] for i in range(7)])
    # Repeated upwind is a binomial convolution: an independent multi-step oracle.
    for nu in (0.2, 0.8, -0.2, -0.8):
        rho = abs(nu)
        direction = 1 if nu > 0 else -1
        current = list(data)
        for step in range(1, 13):
            current = upwind_step(current, nu)
            expected = []
            for i in range(7):
                expected.append(math.fsum(math.comb(step, k) * rho**k
                    * (1.0 - rho)**(step-k) * data[(i-direction*k) % 7]
                    for k in range(step+1)))
            assert_close(current, expected)
            assert abs(math.fsum(current) - math.fsum(data)) < 2e-14
    # Complex exponentials are used only in this check, not in the teaching code.
    mode_checks = 0
    for count in (4, 7, 12):
        for m in range(1, count):
            theta = 2 * math.pi * m / count
            for nu in (-0.8, 0.5):
                factor = 1 - 1j * nu * math.sin(theta)
                current = [math.cos(i * theta) for i in range(count)]
                for step in range(1, 6):
                    current = centered_step(current, nu)
                    expected = [(factor**step * cmath.exp(1j * i * theta)).real
                                for i in range(count)]
                    assert_close(current, expected)
                    mode_checks += 1
    alternating = [1.0, -1.0, 1.0, -1.0]
    for nu in (1.1, -1.1):
        assert_close(upwind_step(alternating, nu), [-1.2, 1.2, -1.2, 1.2])
    for invalid in ((3, 1., 1, 1.), (4, -1., 1, 1.), (4, 1., 0, 1.),
                    (4, 1., 1, float("nan"))):
        try:
            check_grid(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid inputs were accepted")
    return f"PASS: hand steps, cyclic shifts, 48 binomial checks, {mode_checks} modal checks, CFL counterexample, invalid inputs"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    message = self_check()
    nodes, histories, summaries = [], [], []

    def collect(*args, **kwargs):
        values, history, summary = simulate(*args, **kwargs)
        nodes.extend(values)
        histories.extend(history)
        summaries.append(summary)
        return summary

    p = config["pulse"]
    if not (0 < p["half_width"] < min(p["center"], 1-p["center"])):
        raise ValueError("pulse support must stay within the unit interval")
    if p["power"] != 6:
        raise ValueError("this experiment uses the documented sixth-power pulse")
    for speed in p["speeds"]:
        name = "pulse_right" if speed > 0 else "pulse_left"
        profile = lambda x: pulse(x, p["center"], p["half_width"], p["power"])
        initial = [profile(i / p["N"]) for i in range(p["N"])]
        collect(name, "upwind", p["N"], p["T"], p["steps"], speed, initial,
                lambda x, t: profile(x-speed*t))
    p = config["mode"]
    if p["N"] % 4 or not math.isfinite(p["amplitude"]) or p["amplitude"] <= 0:
        raise ValueError("mode experiment needs N divisible by four and amplitude > 0")
    h, dt, nu = check_grid(p["N"], p["T"], p["steps"], p["speed"])
    # Exact integer pattern avoids trigonometric roundoff at nominal zeros.
    initial = [p["amplitude"] * (1, 0, -1, 0)[i % 4] for i in range(p["N"])]
    exact = lambda x, t: p["amplitude"] * math.cos(2*math.pi*(p["N"]//4)*(x-p["speed"]*t))
    for scheme in ("centered", "upwind"):
        factor = math.sqrt(1+nu*nu) if scheme == "centered" else math.hypot(1-abs(nu), abs(nu))
        collect("mode_"+scheme, scheme, p["N"], p["T"], p["steps"], p["speed"],
                initial, exact, amplification=factor)
    p = config["convergence"]
    if p["speed"] <= 0 or not 0 < p["courant"] < 1:
        raise ValueError("convergence experiment requires c > 0 and 0 < courant < 1")
    convergence = []
    previous_error = None
    previous_h = None
    for count in p["grids"]:
        requested_steps = p["T"] * p["speed"] * count / p["courant"]
        steps = round(requested_steps)
        if not math.isclose(steps, requested_steps, rel_tol=0, abs_tol=1e-12):
            raise ValueError("T, N and Courant number must give an integer step count")
        initial = [math.sin(2*math.pi*i/count) for i in range(count)]
        exact = lambda x, t: math.sin(2*math.pi*(x-p["speed"]*t))
        row = collect(f"sine_{count}", "upwind", count, p["T"], steps,
                      p["speed"], initial, exact)
        ratio = previous_error / row["max_error"] if previous_error else ""
        order = math.log(ratio) / math.log(previous_h/row["h"]) if previous_error else ""
        convergence.append({"N": count, "h": row["h"], "dt": row["dt"],
                            "max_error": row["max_error"], "error_ratio": ratio,
                            "observed_order": order})
        previous_error, previous_h = row["max_error"], row["h"]
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("nodes.csv", nodes), ("history.csv", histories),
                           ("summary.csv", summaries), ("convergence.csv", convergence)):
        write_csv(output / filename, rows)
    log = [f"Python {platform.python_version()}", "Solver dependencies: standard library only",
           message, f"Completed {len(summaries)} cases; {len(nodes)} final nodes; {len(histories)} history rows."]
    for row in summaries:
        log.append(f"{row['case']}: max_error={row['max_error']:.9e}; "
                   f"total_drift={row['max_total_drift']:.3e}; "
                   f"norm_ratio={row['final_norm_ratio']:.9e}")
    log.append("All recorded values are from actual time stepping; no timing benchmark is claimed.")
    text = "\n".join(log) + "\n"
    (output / "run.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/advection.json")
    parser.add_argument("--output", type=Path, default=root / "results")
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as stream:
        configuration = json.load(stream)
    run(configuration, arguments.output)
