#!/usr/bin/env python3
"""Burgers cell averages with entropy-aware and deliberately wrong fluxes.

Only Python's standard library is needed. Plotting is a separate entry point.
"""

import argparse
import csv
import json
import math
import platform
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def godunov_flux(left, right):
    if left <= right:
        if left >= 0.0:
            return 0.5 * left * left
        if right <= 0.0:
            return 0.5 * right * right
        return 0.0
    speed = 0.5 * (left + right)
    if speed >= 0.0:
        return 0.5 * left * left
    return 0.5 * right * right


def jump_only_flux(left, right):
    """Counterexample: uses a jump even when the entropy solution is a fan."""
    speed = 0.5 * (left + right)
    if speed >= 0.0:
        return 0.5 * left * left
    return 0.5 * right * right


def exact_average(x_left, x_right, time, a, b):
    """Integrate the exact entropy solution, splitting at wave boundaries."""
    width = x_right - x_left
    if time == 0.0 or a > b:
        position = 0.5 * (a + b) * time
        left_length = max(0.0, min(x_right, position) - x_left)
        return (a * left_length + b * (width - left_length)) / width
    if a == b:
        return a
    left_length = max(0.0, min(x_right, a * time) - x_left)
    right_length = max(0.0, x_right - max(x_left, b * time))
    fan_left = max(x_left, a * time)
    fan_right = min(x_right, b * time)
    fan_integral = 0.0
    if fan_left < fan_right:
        fan_integral = (fan_right - fan_left) * (fan_right + fan_left) / (2.0 * time)
    return (a * left_length + fan_integral + b * right_length) / width


def one_step(values, a, b, dt, h, flux):
    # Each face is computed once, then shared by its two neighboring cells.
    faces = [flux(a, values[0])]
    for i in range(len(values) - 1):
        faces.append(flux(values[i], values[i + 1]))
    faces.append(flux(values[-1], b))
    new = []
    residuals = []
    for i, old in enumerate(values):
        value = old - dt / h * (faces[i + 1] - faces[i])
        new.append(value)
        residuals.append(h * (value - old) - dt * (faces[i] - faces[i + 1]))
    return new, faces, max(abs(r) for r in residuals)


def solve(case, config):
    n, a, b = case['N'], case['a'], case['b']
    if not isinstance(n, int) or n < 4 or n % 2:
        raise ValueError('N must be an even integer, at least 4')
    start, end = config['domain']
    final_time, target = config['final_time'], config['cfl']
    if not (start < 0.0 < end and start == -end and final_time > 0.0):
        raise ValueError('use a symmetric domain containing the initial jump')
    if not (0.0 < target <= 1.0) or not all(math.isfinite(v) for v in (a, b)):
        raise ValueError('finite states and 0 < CFL <= 1 are required')
    h = (end - start) / n
    steps = max(1, math.ceil(final_time * max(abs(a), abs(b)) / (target * h)))
    dt = final_time / steps
    edges = [start + i * h for i in range(n + 1)]
    initial = [exact_average(edges[i], edges[i + 1], 0.0, a, b) for i in range(n)]
    values = initial.copy()
    flux = {'godunov': godunov_flux, 'jump_only': jump_only_flux}[case['method']]
    initial_mass = h * math.fsum(initial)
    exchanges, history = [], []
    max_local = max_balance = max_cfl = 0.0
    previous_local = 0.0
    for step in range(steps + 1):
        time = final_time * step / steps
        exact = [exact_average(edges[i], edges[i + 1], time, a, b) for i in range(n)]
        errors = [values[i] - exact[i] for i in range(n)]
        mass = h * math.fsum(values)
        exchange = math.fsum(exchanges)
        balance = mass - initial_mass - exchange
        current_cfl = dt / h * max(abs(v) for v in values)
        if current_cfl > 1.0 + 1e-13 or not all(math.isfinite(v) for v in values):
            raise ValueError('actual values violate the step condition')
        max_balance = max(max_balance, abs(balance))
        max_cfl = max(max_cfl, current_cfl)
        history.append(dict(case=case['name'], step=step, time=time, mass=mass,
                            boundary_exchange=exchange, mass_balance=balance,
                            minimum=min(values), maximum=max(values), cfl=current_cfl,
                            l1_error=h * math.fsum(abs(e) for e in errors),
                            max_error=max(abs(e) for e in errors),
                            previous_step_balance=previous_local))
        if step < steps:
            values, faces, previous_local = one_step(values, a, b, dt, h, flux)
            exchanges.append(dt * (faces[0] - faces[-1]))
            max_local = max(max_local, previous_local)
    cells = [dict(case=case['name'], i=i, left=edges[i], right=edges[i + 1],
                  initial=initial[i], exact=exact[i], numerical=values[i], error=errors[i])
             for i in range(n)]
    summary = dict(case=case['name'], method=case['method'], N=n, a=a, b=b,
                   h=h, dt=dt, steps=steps, final_time=final_time, mass_initial=initial_mass,
                   mass_final=mass, boundary_exchange=exchange, max_mass_balance=max_balance,
                   max_local_balance=max_local, max_cfl=max_cfl,
                   l1_error=history[-1]['l1_error'], max_error=history[-1]['max_error'],
                   minimum=min(values), maximum=max(values))
    return cells, history, summary


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'config/burgers.json')
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cells, history, summary = [], [], []
    for case in config['cases']:
        case_cells, case_history, case_summary = solve(case, config)
        cells.extend(case_cells)
        history.extend(case_history)
        summary.append(case_summary)
    convergence = []
    by_name = {r['case']: r for r in summary}
    for n in config['refinements']:
        convergence.append(dict(N=n, shock_l1=by_name[f'shock_{n}']['l1_error'],
                                rarefaction_l1=by_name[f'rarefaction_{n}']['l1_error'],
                                jump_only_l1=by_name[f'jump_only_{n}']['l1_error']))
    jumps = []
    for a, b in [(2.0, 0.0), (0.0, -2.0), (1.0, -1.0), (-1.0, 1.0)]:
        speed = (a + b) / 2.0
        jumps.append(dict(a=a, b=b, speed=speed,
                          rh_residual=speed * (b - a) - (b * b - a * a) / 2.0,
                          entropy=(b**3 - a**3) / 3.0 - speed * (b * b - a * a) / 2.0,
                          entropy_factored=(b - a)**3 / 12.0,
                          left_relative=a - speed, right_relative=b - speed))
    characteristics = [dict(time=t, xi=0.0, position=0.0, value=0.0,
                            jacobian=1.0 - t, slope=-1.0 / (1.0 - t))
                       for t in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]]
    hand = []
    for method, flux in [('godunov', godunov_flux), ('jump_only', jump_only_flux)]:
        old = [-1.0, -1.0, 1.0, 1.0]
        new, faces, _ = one_step(old, -1.0, 1.0, 0.25, 1.0, flux)
        for i in range(4):
            hand.append(dict(method=method, i=i, old=old[i], new=new[i], h=1.0, dt=0.25,
                             left_flux=faces[i], right_flux=faces[i + 1]))
    outputs = dict(cells=cells, history=history, summary=summary, convergence=convergence,
                   jumps=jumps, characteristics=characteristics, hand=hand)
    # Validation and all calculations finish before any output file is written.
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(args.output / (name + '.csv'), rows)
    report = [f'Python {platform.python_version()}; standard library only',
              f'cases={len(summary)}; updates={sum(r["steps"] for r in summary)}',
              *(f'{name}.csv: {len(rows)} data rows' for name, rows in outputs.items()),
              f'max_mass_balance={max(r["max_mass_balance"] for r in summary):.17g}',
              f'max_local_balance={max(r["max_local_balance"] for r in summary):.17g}']
    text = '\n'.join(report) + '\n'
    (args.output / 'run.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
