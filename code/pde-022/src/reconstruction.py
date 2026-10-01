#!/usr/bin/env python3
"""Periodic positive-speed advection with three choices of reconstruction slope."""

import argparse
import csv
import json
import math
import platform
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def minmod(left, right):
    if left > 0.0 and right > 0.0:
        return min(left, right)
    if left < 0.0 and right < 0.0:
        return max(left, right)
    return 0.0


def reconstruct(values, h, method):
    slopes = []
    count = len(values)
    for i in range(count):
        left = (values[i] - values[i - 1]) / h
        right = (values[(i + 1) % count] - values[i]) / h
        if method == 'upwind':
            slopes.append(0.0)
        elif method == 'centered':
            slopes.append((values[(i + 1) % count] - values[i - 1]) / (2.0 * h))
        elif method == 'minmod':
            slopes.append(minmod(left, right))
        else:
            raise ValueError('unknown slope method')
    return slopes


def advance(values, h, dt, c, method):
    if not (h > 0.0 and dt > 0.0 and c > 0.0 and 0.0 < c * dt / h <= 1.0 + 1e-14):
        raise ValueError('positive h, dt, c and CFL <= 1 are required')
    slopes = reconstruct(values, h, method)
    fluxes = []
    for i in range(len(values)):
        # Time average of the translated line at the right face of cell i.
        mean_value = values[i] + 0.5 * (h - c * dt) * slopes[i]
        fluxes.append(c * mean_value)
    new = []
    local_balance = 0.0
    for i, old in enumerate(values):
        value = old - dt / h * (fluxes[i] - fluxes[i - 1])
        if not math.isfinite(value):
            raise ValueError('non-finite updated cell average')
        new.append(value)
        residual = h * (value - old) - dt * (fluxes[i - 1] - fluxes[i])
        local_balance = max(local_balance, abs(residual))
    return new, slopes, fluxes, local_balance


def exact_averages(profile, count, shift, config):
    h = 1.0 / count
    values = []
    for i in range(count):
        if profile == 'cosine':
            amplitude = config['cosine']['amplitude']
            offset = config['cosine']['offset']
            average = offset + amplitude * math.sin(math.pi * h) / (math.pi * h) * math.cos(
                2.0 * math.pi * ((i + 0.5) * h - shift))
        elif profile == 'pulse':
            left, right = i * h - shift, (i + 1) * h - shift
            start, end = config['pulse']
            integral = 0.0
            for period in range(math.floor(left) - 1, math.ceil(right) + 1):
                integral += max(0.0, min(right, period + end) - max(left, period + start))
            average = integral / h
        elif profile == 'constant':
            average = config['constant']
        else:
            raise ValueError('unknown profile')
        values.append(average)
    return values


def total_variation(values):
    return math.fsum(abs(value - values[i - 1]) for i, value in enumerate(values))


def solve(case, config):
    n, steps = case['N'], case['steps']
    if not isinstance(n, int) or n < 4 or not isinstance(steps, int) or steps < 1:
        raise ValueError('N >= 4 and steps >= 1 must be integers')
    c, final_time = config['c'], config['final_time']
    h, dt = 1.0 / n, final_time / steps
    if not (math.isfinite(c) and math.isfinite(final_time) and c > 0 and final_time > 0
            and c * dt / h <= 1.0 + 1e-14):
        raise ValueError('positive finite speed/time and CFL <= 1 are required')
    initial = exact_averages(case['profile'], n, 0.0, config)
    values = initial.copy()
    initial_mass, initial_tv = h * math.fsum(values), total_variation(values)
    history = []
    max_drift = max_local = max_tv_increase = 0.0
    previous_local, previous_tv = 0.0, initial_tv
    for step in range(steps + 1):
        time = final_time * step / steps
        exact = exact_averages(case['profile'], n, c * time, config)
        errors = [value - target for value, target in zip(values, exact)]
        mass, tv = h * math.fsum(values), total_variation(values)
        drift = mass - initial_mass
        tv_change = tv - previous_tv
        max_drift = max(max_drift, abs(drift))
        max_local = max(max_local, previous_local)
        max_tv_increase = max(max_tv_increase, tv_change)
        history.append(dict(case=case['name'], step=step, time=time, mass=mass,
                            mass_drift=drift, minimum=min(values), maximum=max(values),
                            tv=tv, tv_change=tv_change,
                            l1_error=h * math.fsum(abs(e) for e in errors),
                            max_error=max(abs(e) for e in errors), previous_step_balance=previous_local))
        if step < steps:
            previous_tv = tv
            values, _, _, previous_local = advance(values, h, dt, c, case['method'])
    cells = [dict(case=case['name'], i=i, left=i * h, right=(i + 1) * h,
                  initial=initial[i], exact=exact[i], numerical=values[i], error=errors[i]) for i in range(n)]
    summary = dict(case=case['name'], method=case['method'], profile=case['profile'], N=n,
                   steps=steps, h=h, dt=dt, c=c, nu=c * dt / h, final_time=final_time,
                   mass_initial=initial_mass, mass_final=mass, max_mass_drift=max_drift,
                   max_local_balance=max_local, tv_initial=initial_tv, tv_final=tv,
                   max_tv_increase=max_tv_increase, minimum=min(values), maximum=max(values),
                   l1_error=history[-1]['l1_error'], max_error=history[-1]['max_error'])
    return cells, history, summary


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'config/advection.json')
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cells, history, summary = [], [], []
    for case in config['cases']:
        case_cells, case_history, case_summary = solve(case, config)
        cells.extend(case_cells)
        history.extend(case_history)
        summary.append(case_summary)
    by_name = {row['case']: row for row in summary}
    convergence = []
    for method in ['upwind', 'centered', 'minmod']:
        previous = None
        for count in config['refinements']:
            row = by_name[f'cosine_{method}_{count}']
            l1_order = max_order = ''
            if previous is not None:
                l1_order = math.log2(previous['l1_error'] / row['l1_error'])
                max_order = math.log2(previous['max_error'] / row['max_error'])
            convergence.append(dict(method=method, N=count, h=row['h'], l1_error=row['l1_error'],
                                    max_error=row['max_error'], l1_order=l1_order, max_order=max_order))
            previous = row
    hand = []
    old = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    for method in ['upwind', 'centered', 'minmod']:
        new, slopes, fluxes, _ = advance(old, 1.0, 0.5, 1.0, method)
        for i in range(6):
            hand.append(dict(method=method, i=i, old=old[i], slope=slopes[i],
                             left_flux=fluxes[i - 1], right_flux=fluxes[i], new=new[i],
                             h=1.0, dt=0.5, tv_old=total_variation(old), tv_new=total_variation(new)))
    reconstruction = []
    for name, stencil in [('corner', [0., 0., 1.]), ('rising', [0., 1., 3.]),
                          ('falling', [3., 1., 0.]), ('peak', [0., 1., 0.]),
                          ('affine', [0., 1., 2.]), ('valley', [0., -1., 0.])]:
        for method in ['upwind', 'centered', 'minmod']:
            slope = reconstruct(stencil, 1.0, method)[1]
            reconstruction.append(dict(stencil=name, method=method, previous=stencil[0],
                                       average=stencil[1], following=stencil[2], h=1.0, slope=slope,
                                       left_value=stencil[1] - .5 * slope,
                                       right_value=stencil[1] + .5 * slope))
    outputs = dict(cells=cells, history=history, summary=summary, convergence=convergence,
                   hand=hand, reconstruction=reconstruction)
    # Complete validation and computation before creating result files.
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(args.output / (name + '.csv'), rows)
    report = [f'Python {platform.python_version()}; standard library only',
              f'cases={len(summary)}; updates={sum(row["steps"] for row in summary)}',
              *(f'{name}.csv: {len(rows)} data rows' for name, rows in outputs.items()),
              f'max_mass_drift={max(row["max_mass_drift"] for row in summary):.17g}',
              f'max_local_balance={max(row["max_local_balance"] for row in summary):.17g}']
    text = '\n'.join(report) + '\n'
    (args.output / 'run.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
