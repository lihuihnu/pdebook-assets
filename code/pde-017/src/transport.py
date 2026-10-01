#!/usr/bin/env python3
"""Periodic advection: explicit updates, modal diagnostics, and saved evidence."""

import argparse
import csv
import json
import math
import platform
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def upwind_step(current, courant):
    count = len(current)
    following = [0.0] * count
    weight = abs(courant)
    direction = 1 if courant >= 0.0 else -1
    for i in range(count):
        upstream = (i - direction) % count
        following[i] = (1.0 - weight) * current[i] + weight * current[upstream]
    return following


def lax_wendroff_step(current, courant):
    count = len(current)
    following = [0.0] * count
    for i in range(count):
        left = current[(i - 1) % count]
        center = current[i]
        right = current[(i + 1) % count]
        following[i] = (center - 0.5 * courant * (right - left)
                        + 0.5 * courant**2 * (right - 2.0 * center + left))
    return following


STEPS = {"upwind": upwind_step, "lax_wendroff": lax_wendroff_step}


def response(scheme, courant, theta):
    sine = 0.0 if theta == math.pi else math.sin(theta)
    cosine = math.cos(theta)
    if scheme == "upwind":
        p = 1 - abs(courant) + abs(courant) * cosine
    else:
        p = 1 - courant**2 * (1 - cosine)
    q = courant * sine
    radius = math.hypot(p, q)
    phase = math.atan2(q, p) if radius > 0 else None
    return radius, phase


def initial_value(x, modes):
    return math.fsum(amplitude * math.cos(2 * math.pi * mode * x)
                     for mode, amplitude in modes)


def project(values, mode):
    count = len(values)
    theta = 2 * math.pi * mode / count
    a = 2 * math.fsum(value * math.cos(i * theta) for i, value in enumerate(values)) / count
    b = 2 * math.fsum(value * math.sin(i * theta) for i, value in enumerate(values)) / count
    return math.hypot(a, b), math.atan2(b, a)


def validate(case):
    count, steps = case["N"], case["steps"]
    if type(count) is not int or count < 4 or type(steps) is not int or steps < 1:
        raise ValueError("N>=4 and steps>=1 must be integers")
    if not math.isfinite(case["c"]) or not math.isfinite(case["T"]) or case["T"] <= 0:
        raise ValueError("finite velocity and positive finite T required")
    if case["scheme"] not in STEPS or not case["modes"]:
        raise ValueError("unknown scheme or missing modes")
    seen = set()
    for mode, amplitude in case["modes"]:
        if type(mode) is not int or not 1 <= mode < count/2 or mode in seen:
            raise ValueError("distinct nondegenerate integer modes required")
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("positive finite modal amplitudes required")
        seen.add(mode)
    courant = case["c"] * case["T"] * count / steps
    if not math.isfinite(courant) or abs(courant) > 1:
        raise ValueError("stable production runs require abs(courant)<=1")
    return courant


def simulate(case):
    nu = validate(case)
    count, steps, terminal = case["N"], case["steps"], case["T"]
    dt, h = terminal / steps, 1 / count
    initial = [initial_value(i*h, case["modes"]) for i in range(count)]
    current = initial[:]
    histories, modal_history, final_modes = [], [], []
    phase_state = {}
    for mode, amplitude in case["modes"]:
        observed, angle = project(initial, mode)
        phase_state[mode] = [angle, 0.0, observed]
    initial_total = h * math.fsum(initial)
    initial_norm = math.sqrt(h * math.fsum(v*v for v in initial))
    largest_drift, largest_prediction_error, largest_norm_increase = 0.0, 0.0, 0.0
    largest_phase_error = 0.0
    for n in range(steps+1):
        if not all(math.isfinite(value) for value in current):
            raise ArithmeticError("nonfinite node value")
        t = n * dt
        exact = [initial_value(i*h - case["c"]*t, case["modes"]) for i in range(count)]
        error = [a-b for a, b in zip(current, exact)]
        total = h * math.fsum(current)
        norm = math.sqrt(h * math.fsum(v*v for v in current))
        largest_drift = max(largest_drift, abs(total-initial_total))
        largest_norm_increase = max(largest_norm_increase, norm-initial_norm)
        histories.append({"case": case["id"], "step": n, "time": t,
                          "minimum": min(current), "maximum": max(current), "total": total,
                          "norm_2h": norm, "error_max": max(abs(v) for v in error),
                          "error_2h": math.sqrt(h * math.fsum(v*v for v in error))})
        for mode, amplitude in case["modes"]:
            radius, phi = response(case["scheme"], nu, 2*math.pi*mode/count)
            measured, angle = project(current, mode)
            previous, unwrapped, reference = phase_state[mode]
            # All configured modes remain resolved in amplitude; fail rather than
            # attach a plausible phase to roundoff after a mode disappears.
            if measured <= 1e-12 * reference or phi is None:
                raise ArithmeticError("modal phase is not resolved")
            if n:
                unwrapped += math.remainder(angle-previous, 2*math.pi)
            phase_state[mode] = [angle, unwrapped, reference]
            ratio = measured/reference
            prediction = radius**n
            discrepancy = abs(ratio-prediction)
            largest_prediction_error = max(largest_prediction_error, discrepancy)
            largest_phase_error = max(largest_phase_error, abs(unwrapped-n*phi))
            row = {"case": case["id"], "step": n, "mode": mode, "time": t,
                   "amplitude_ratio": ratio, "predicted_ratio": prediction,
                   "phase": unwrapped, "predicted_phase": n*phi,
                   "phase_error": unwrapped-2*math.pi*mode*case["c"]*t}
            if case["group"] in ("modal", "waveform"):
                modal_history.append(row)
            if n == steps:
                final_modes.append(row)
            if discrepancy > 2e-11 or abs(unwrapped-n*phi) > 2e-8:
                raise ArithmeticError("modal prediction disagreement")
        if n < steps:
            current = STEPS[case["scheme"]](current, nu)
    if largest_drift > 2e-13 or largest_norm_increase > 2e-13:
        raise ArithmeticError("conservation or norm bound failed")
    nodes = [{"case": case["id"], "i": i, "x": i*h, "initial": initial[i],
              "numerical": current[i], "exact": exact[i]} for i in range(count)]
    summary = {"case": case["id"], "group": case["group"], "scheme": case["scheme"],
               "N": count, "steps": steps, "c": case["c"], "T": terminal, "dt": dt,
               "courant": nu, "error_max": histories[-1]["error_max"],
               "error_2h": histories[-1]["error_2h"], "max_total_drift": largest_drift,
               "max_norm_increase": largest_norm_increase,
               "max_amplitude_prediction_error": largest_prediction_error,
               "max_phase_prediction_error": largest_phase_error}
    return nodes, histories, modal_history, final_modes, summary


def self_check():
    import cmath
    values = [0.0, 1.0, 0.0, 0.0, 0.0]
    assert all(abs(a-b) < 1e-15 for a, b in zip(lax_wendroff_step(values, .8), [-.08, .36, .72, 0, 0]))
    assert values == [0, 1, 0, 0, 0]
    for step in STEPS.values():
        assert step(values, 1) == values[-1:] + values[:-1]
        assert step(values, -1) == values[1:] + values[:1]
        assert step(values, 0) == values
        assert max(abs(v) for v in step([1, -1]*4, 1.1)) > 1
    assert upwind_step([1, -1]*4, .5) == [0]*8
    # Independent complex symbol checks include negative velocity and mixed waves.
    checks = 0
    for nu in (-.8, -.5, .5, .8, 1):
        for scheme, step in STEPS.items():
            count, theta = 16, 2*math.pi*3/16
            current = [math.cos(i*theta) for i in range(count)]
            z = cmath.exp(-1j*theta)
            symbol = (1-abs(nu)+abs(nu)*z**(1 if nu >= 0 else -1)
                      if scheme == "upwind" else 1-.5*nu*(1/z-z)+.5*nu**2*(1/z-2+z))
            for n in range(1, 11):
                current = step(current, nu)
                predicted = [(symbol**n * cmath.exp(1j*i*theta)).real for i in range(count)]
                assert max(abs(a-b) for a,b in zip(current,predicted)) < 2e-14
                checks += 1
    valid = {"id":"test", "scheme":"upwind", "N":40, "steps":50,
             "c":1, "T":1, "modes":[[1,1]], "group":"test"}
    for change in ({"N":0},{"steps":0},{"c":float("nan")},{"T":-1},{"c":2},
                   {"scheme":"unknown"},{"modes":[[20,1]]},{"modes":[[1,0]]}):
        try:
            validate(dict(valid, **change))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration accepted")
    return f"PASS: hand step, shifts/stationary, non-mutation, CFL counterexamples, alternating-mode annihilation, {checks} complex-symbol checks, invalid inputs"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE / "config/transport.json")
    parser.add_argument("--output", type=Path, default=BASE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = config["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    log = [f"Python {platform.python_version()}; standard library only", self_check()]
    nodes, history, mode_history, modes, summary = [], [], [], [], []
    for case in cases:
        result = simulate(case)
        for target, rows in zip((nodes, history, mode_history, modes), result[:4]):
            target.extend(rows)
        summary.append(result[4])
        log.append(f"{case['id']}: max_error={result[4]['error_max']:.9e}; l2_error={result[4]['error_2h']:.9e}; total_drift={result[4]['max_total_drift']:.3e}")
    scan = []
    intervals = config["scan_intervals"]
    for nu in config["scan_courant"]:
        for scheme in STEPS:
            for j in range(intervals+1):
                theta = math.pi*j/intervals
                radius, phi = response(scheme, nu, theta)
                # Endpoints are degenerate modes; only the zero-wavenumber limit
                # has a defined speed ratio here. Leave the pi endpoint blank.
                speed = 1.0 if j == 0 else (phi/(nu*theta) if j < intervals and phi is not None else "")
                scan.append({"scheme":scheme, "courant":nu, "theta_over_pi":j/intervals,
                             "R":radius, "phase":phi if phi is not None else "", "speed_ratio":speed})
    convergence = []
    for scheme in STEPS:
        previous = None
        for row in sorted((r for r in summary if r["group"] in ("waveform","refinement") and r["scheme"] == scheme), key=lambda r:r["N"]):
            error = row["error_max"]
            ratio = previous/error if previous is not None else ""
            convergence.append({"scheme":scheme, "N":row["N"], "h":1/row["N"],
                                "dt":row["dt"], "error_max":error, "error_2h":row["error_2h"],
                                "ratio":ratio, "order":math.log2(ratio) if ratio else ""})
            previous = error
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("nodes",nodes),("history",history),("mode_history",mode_history),
                       ("modes",modes),("summary",summary),("scan",scan),("convergence",convergence)):
        write_csv(args.output / (name+".csv"), rows)
    log.append(f"Completed {len(cases)} cases; {len(nodes)} final nodes; {len(history)} time records; {len(mode_history)} modal time records; {len(scan)} analytic scan rows.")
    text = "\n".join(log)+"\n"
    (args.output / "run.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
