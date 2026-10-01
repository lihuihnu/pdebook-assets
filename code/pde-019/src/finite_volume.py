#!/usr/bin/env python3
"""Shared numerical fluxes for periodic constant-speed advection (stdlib only)."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lax_friedrichs_flux(left, right, c, h, dt):
    return 0.5 * c * (left + right) - 0.5 * h / dt * (right - left)


def periodic_step(values, c, h, dt, numerical_flux):
    count = len(values)
    fluxes = [0.0] * (count + 1)
    fluxes[0] = numerical_flux(values[-1], values[0], c, h, dt)
    for j in range(1, count):
        fluxes[j] = numerical_flux(values[j - 1], values[j], c, h, dt)
    fluxes[count] = fluxes[0]
    updated = []
    for i in range(count):
        change = dt / h * (fluxes[i + 1] - fluxes[i])
        updated.append(values[i] - change)
    return updated, fluxes


def central_flux(left, right, c, h, dt):
    return 0.5 * c * (left + right)


def pulse_integral(x, start, end):
    """Integral from zero to x of the unit-periodic indicator of [start, end)."""
    periods = math.floor(x)
    remainder = x - periods
    return periods * (end - start) + max(0.0, min(remainder, end) - start)


def exact_averages(case, profiles, time):
    count, c = case['N'], case['c']
    h = 1.0 / count
    profile = profiles[case['profile']]
    values = []
    for i in range(count):
        if case['profile'] == 'cosine':
            center = (i + 0.5) * h
            average = profile['mean'] + profile['amplitude'] * math.sin(math.pi * h) / (math.pi * h) * math.cos(2 * math.pi * (center - c * time))
        elif case['profile'] == 'pulse':
            left = i * h - c * time
            right = (i + 1) * h - c * time
            average = (pulse_integral(right, profile['start'], profile['end']) - pulse_integral(left, profile['start'], profile['end'])) / h
        elif case['profile'] == 'constant':
            average = profile['value']
        else:
            raise ValueError('Unknown initial profile')
        values.append(average)
    return values


def validate(config):
    names = set()
    for case in config['cases']:
        if case['id'] in names:
            raise ValueError('Duplicate case ID')
        names.add(case['id'])
        for key, minimum in [('N', 2), ('steps', 1)]:
            if type(case[key]) is not int or case[key] < minimum:
                raise ValueError(key + ' must be an integer at least ' + str(minimum))
        if not math.isfinite(case['c']) or not math.isfinite(case['T']) or case['T'] <= 0:
            raise ValueError('c must be finite and T must be positive and finite')
        if case['flux'] not in ('central', 'lf') or case['profile'] not in config['profiles']:
            raise ValueError('Unknown flux or profile')
        nu = case['c'] * case['T'] / case['steps'] * case['N']
        if case['flux'] == 'lf' and abs(nu) > 1 + 1e-14 and not case.get('allow_cfl_violation', False):
            raise ValueError('LF requires abs(nu) <= 1 unless an unstable control is explicitly requested')
    for name, profile in config['profiles'].items():
        if name not in ('cosine', 'pulse', 'constant') or not all(math.isfinite(x) for x in profile.values()):
            raise ValueError('Invalid profile parameters')
    pulse = config['profiles']['pulse']
    if not 0 <= pulse['start'] < pulse['end'] <= 1:
        raise ValueError('Pulse must lie inside one period')


def write_csv(directory, name, rows):
    with (directory / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, profiles, tables):
    count, steps, c = case['N'], case['steps'], case['c']
    h, dt = 1.0 / count, case['T'] / steps
    numerical_flux = central_flux if case['flux'] == 'central' else lax_friedrichs_flux
    values = exact_averages(case, profiles, 0.0)
    initial = values[:]
    initial_total = h * math.fsum(initial)
    max_drift = max_balance = 0.0
    for n in range(steps + 1):
        time = n * dt
        exact = exact_averages(case, profiles, time)
        errors = [values[i] - exact[i] for i in range(count)]
        total = h * math.fsum(values)
        max_error = max(abs(e) for e in errors)
        l1_error = h * math.fsum(abs(e) for e in errors)
        max_drift = max(max_drift, abs(total - initial_total))
        tables['history'].append(dict(case=case['id'], step=n, time=time, total=total,
                                     mass_drift=total-initial_total, minimum=min(values),
                                     maximum=max(values), max_error=max_error, l1_error=l1_error))
        if n == steps:
            break
        updated, fluxes = periodic_step(values, c, h, dt, numerical_flux)
        if not all(math.isfinite(v) for v in updated + fluxes):
            raise ArithmeticError('Non-finite result in ' + case['id'])
        for i in range(count):
            mass_change = h * (updated[i] - values[i])
            net_inflow = dt * (fluxes[i] - fluxes[i+1])
            residual = mass_change - net_inflow
            max_balance = max(max_balance, abs(residual))
            if n in (0, steps-1):
                tables['updates'].append(dict(case=case['id'], step=n, i=i, width=h,
                                              old=values[i], new=updated[i], left_flux=fluxes[i],
                                              right_flux=fluxes[i+1], mass_change=mass_change,
                                              net_inflow=net_inflow, residual=residual))
        if n in (0, steps-1):
            for j in range(count + 1):
                tables['faces'].append(dict(case=case['id'], step=n, time=time, j=j,
                                            x=j*h, flux=fluxes[j]))
        values = updated
    for i in range(count):
        tables['cells'].append(dict(case=case['id'], i=i, left=i*h, right=(i+1)*h,
                                   center=(i+0.5)*h, width=h, initial=initial[i],
                                   exact=exact[i], numerical=values[i], error=errors[i]))
    result = dict(case=case['id'], profile=case['profile'], flux=case['flux'], N=count,
                  c=c, T=case['T'], steps=steps, h=h, dt=dt, nu=c*dt/h,
                  initial_total=initial_total, final_total=total, max_mass_drift=max_drift,
                  max_balance_residual=max_balance, minimum=min(values), maximum=max(values),
                  max_error=max_error, l1_error=l1_error)
    tables['summary'].append(result)
    return result


def algebra_examples(tables):
    values = [1.0, 2.0, 3.0, 2.0]
    updated, fluxes = periodic_step(values, 1.0, 0.25, 0.125, lax_friedrichs_flux)
    for i in range(4):
        tables['hand'].append(dict(i=i, old=values[i], left_flux=fluxes[i],
                                  right_flux=fluxes[i+1], new=updated[i]))
    examples = [
        ('open_nonuniform', [0.2, 0.3, 0.5], [1.0, 2.0, 0.0], [0.5, 0.2, 0.4, 0.1], 0.1),
        ('periodic_endpoint_defect', [0.25]*4, values, fluxes[:-1] + [fluxes[-1]+0.1], 0.125),
    ]
    for name, widths, old, faces, dt in examples:
        new = [old[i] + dt / widths[i] * (faces[i]-faces[i+1]) for i in range(len(old))]
        before = math.fsum(w*v for w, v in zip(widths, old))
        after = math.fsum(w*v for w, v in zip(widths, new))
        for i in range(len(old)):
            tables['boundary_examples'].append(dict(example=name, i=i, width=widths[i], dt=dt,
                                                     old=old[i], left_flux=faces[i], right_flux=faces[i+1],
                                                     new=new[i], initial_total=before, final_total=after,
                                                     boundary_change=dt*(faces[0]-faces[-1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/fluxes.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate(config)
    tables = {name: [] for name in ['cells', 'history', 'faces', 'updates', 'summary',
                                    'convergence', 'hand', 'boundary_examples']}
    previous = None
    for case in config['cases']:
        result = run_case(case, config['profiles'], tables)
        print(f"{case['id']}: nu={result['nu']:.6g}, max_error={result['max_error']:.12e}, "
              f"mass_drift={result['max_mass_drift']:.3e}, range=[{result['minimum']:.9f}, {result['maximum']:.9f}]")
        if case.get('convergence', False):
            ratio = previous / result['max_error'] if previous is not None else ''
            order = math.log2(ratio) if ratio != '' else ''
            tables['convergence'].append(dict(N=case['N'], h=result['h'], dt=result['dt'],
                                               max_error=result['max_error'], previous_ratio=ratio,
                                               observed_order=order))
            previous = result['max_error']
    algebra_examples(tables)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output, name, rows)
    print('rows: ' + ', '.join(name + '=' + str(len(rows)) for name, rows in tables.items()))
    print('max_mass_drift:', max(row['max_mass_drift'] for row in tables['summary']))
    print('max_balance_residual:', max(row['max_balance_residual'] for row in tables['summary']))


if __name__ == '__main__':
    main()
