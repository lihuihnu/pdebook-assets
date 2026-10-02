#!/usr/bin/env python3
"""Check Poisson weak integrals by composite midpoint quadrature.

This script evaluates prescribed functions; it does not solve a finite element
system. The standard library is sufficient. Plotting is a separate operation.
"""

import argparse
import csv
import json
import math
import platform
from pathlib import Path


def weak_integrals(derivative, test, test_derivative, m):
    """Use m midpoint cells on each side of the corner x=1/2; f=1."""
    if type(m) is not int or m <= 0:
        raise ValueError("m must be a positive integer")
    h = 0.5 / m
    left = 0.0
    right = 0.0
    for start in (0.0, 0.5):
        for j in range(m):
            x = start + (j + 0.5) * h
            left += h * derivative(x) * test_derivative(x)
            right += h * test(x)
    return left, right, left - right


def exact_value(x):
    return x * (1.0 - x) / 2.0


def exact_derivative(x):
    return 0.5 - x


def tent_derivative(x, slope):
    if x == 0.5:
        raise ValueError("the tent derivative is undefined at x=1/2")
    return slope if x < 0.5 else -slope


def w_value(x):
    return min(x, 1.0 - x) / 4.0


def w_derivative(x):
    return tent_derivative(x, 0.25)


def z_value(x):
    return min(x, 1.0 - x) / 3.0


def z_derivative(x):
    return tent_derivative(x, 1.0 / 3.0)


def v1(x):
    return x * (1.0 - x)


def dv1(x):
    return 1.0 - 2.0 * x


def v2(x):
    return x * x * (1.0 - x) ** 2


def dv2(x):
    return 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x)


def odd_test(x):
    return x * (1.0 - x) * (1.0 - 2.0 * x)


def odd_derivative(x):
    return 1.0 - 6.0 * x + 6.0 * x * x


def local_test(x):
    if 0.125 < x < 0.375:
        return (x - 0.125) ** 2 * (0.375 - x) ** 2
    return 0.0


def local_derivative(x):
    if 0.125 < x < 0.375:
        return 2.0 * (x - 0.125) * (0.375 - x) * (0.5 - 2.0 * x)
    return 0.0


CANDIDATES = {
    "u": (exact_value, exact_derivative),
    "w": (w_value, w_derivative),
    "z": (z_value, z_derivative),
}
TESTS = {
    "v1": (v1, dv1),
    "v2": (v2, dv2),
    "odd": (odd_test, odd_derivative),
    "local": (local_test, local_derivative),
}


def reference_integrals(candidate, test_name):
    # These follow from elementary antiderivatives, not from quadrature.
    areas = {"v1": 1.0 / 6, "v2": 1.0 / 30, "odd": 0.0,
             "local": (0.375 - 0.125) ** 5 / 30}
    right = areas[test_name]
    if candidate == "u":
        left = right
    else:
        slope = 0.25 if candidate == "w" else 1.0 / 3.0
        left = 2.0 * slope * TESTS[test_name][0](0.5)
    return left, right, left - right


def energy_integrals(value, derivative, m):
    h = 0.5 / m
    energy = 0.0
    slope_error_squared = 0.0
    for start in (0.0, 0.5):
        for j in range(m):
            x = start + (j + 0.5) * h
            slope = derivative(x)
            energy += h * (0.5 * slope * slope - value(x))
            slope_error_squared += h * (slope - exact_derivative(x)) ** 2
    return energy, slope_error_squared


def self_check():
    one = lambda x: 1.0
    zero = lambda x: 0.0
    assert weak_integrals(one, zero, one, 4) == (1.0, 0.0, 1.0)
    assert weak_integrals(w_derivative, v1, dv1, 4)[0] == 0.125
    for invalid in (0, -1, 1.5, True):
        try:
            weak_integrals(one, one, one, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid m was accepted")
    for derivative in (w_derivative, z_derivative):
        try:
            derivative(0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("corner derivative was assigned a value")


def csv_output(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ArithmeticError("non-finite result")
            writer.writerow({key: format(value, ".17g") if isinstance(value, float)
                             else value for key, value in row.items()})


def run(config, output):
    levels = config["midpoint_cells_per_half"]
    samples = config["profile_intervals"]
    if (not isinstance(levels, list) or not levels
            or any(type(m) is not int or m < 4 or m % 4 for m in levels)
            or levels != sorted(set(levels))):
        raise ValueError("levels must be distinct increasing multiples of four")
    if type(samples) is not int or samples < 4 or samples % 2:
        raise ValueError("profile_intervals must be even and at least four")
    self_check()
    integrals = []
    references = []
    energies = []
    for name, (value, derivative) in CANDIDATES.items():
        for test_name, (test, dtest) in TESTS.items():
            ref_left, ref_right, ref_residual = reference_integrals(name, test_name)
            references.append(dict(candidate=name, test=test_name, left=ref_left,
                                   right=ref_right, residual=ref_residual))
            for m in levels:
                left, right, residual = weak_integrals(derivative, test, dtest, m)
                integrals.append(dict(candidate=name, test=test_name, m=m,
                                      h=0.5/m, left=left, right=right,
                                      residual=residual, exact_residual=ref_residual,
                                      quadrature_error=abs(residual-ref_residual)))
        if name == "u":
            energy_exact, slope_exact = -1.0/24, 0.0
        else:
            a = 0.25 if name == "w" else 1.0/3
            energy_exact = a*a/2 - a/4
            slope_exact = a*a - a/2 + 1.0/12
        for m in levels:
            energy, slope_squared = energy_integrals(value, derivative, m)
            energies.append(dict(candidate=name, m=m, energy=energy,
                                 exact_energy=energy_exact,
                                 slope_error_squared=slope_squared,
                                 exact_slope_error_squared=slope_exact))
    profiles = []
    for j in range(samples + 1):
        x = j / samples
        profiles.append(dict(x=x, u=exact_value(x), w=w_value(x), z=z_value(x)))
    slopes = []
    for side, start in (("left", 0.0), ("right", 0.5)):
        sign = 1 if side == "left" else -1
        for j in range(samples // 2 + 1):
            x = start + j / samples
            # At x=1/2 these are one-sided limits, labelled by side.
            slopes.append(dict(side=side, x=x, u_prime=0.5-x,
                               w_prime=sign/4, z_prime=sign/3))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("integrals", integrals), ("exact_integrals", references),
                       ("energies", energies), ("profiles", profiles),
                       ("slopes", slopes)):
        csv_output(output / (name + ".csv"), rows)
    log = (f"Python {platform.python_version()}; standard library only\n"
           "Problem: -u''=1 on (0,1); u(0)=u(1)=0; prescribed u,w,z\n"
           f"Quadrature: {len(integrals)} candidate/test/level combinations\n"
           f"Energy checks: {len(energies)}; exact integral pairs: {len(references)}\n"
           f"Profiles: {len(profiles)}; one-sided derivative rows: {len(slopes)}\n"
           "Midpoint partitions split at 1/2; local-test endpoints are aligned\n"
           "Self-check: constant integral, linear half-integral, invalid m, corner rejection passed\n"
           "Finite selected tests do not prove the all-test weak equation\n")
    (output / "run.txt").write_text(log, encoding="utf-8")
    print(log, end="")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/quadrature.json")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
