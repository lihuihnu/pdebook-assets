#!/usr/bin/env python3
"""Upwind cell averages, exact remapping and periodic advection (stdlib only)."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def upwind_flux(left, right, c, h, dt):
    if c >= 0.0:
        return c * left
    return c * right


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


def lax_friedrichs_flux(left, right, c, h, dt):
    return 0.5 * c * (left + right) - 0.5 * h / dt * (right - left)


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
        if case['flux'] not in ('upwind', 'lf') or case['profile'] not in config['profiles']:
            raise ValueError('Unknown flux or profile')
        nu = case['c'] * case['T'] / case['steps'] * case['N']
        if abs(nu) > 1 + 1e-14:
            raise ValueError('Propagation cases require abs(nu) <= 1; long shifts are isolated algebra comparisons')
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
    numerical_flux = upwind_flux if case['flux'] == 'upwind' else lax_friedrichs_flux
    values = exact_averages(case, profiles, 0.0)
    initial = values[:]
    initial_total = h * math.fsum(initial)
    max_drift = max_balance = last_balance = 0.0
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
                                     maximum=max(values), max_error=max_error, l1_error=l1_error,
                                     previous_step_balance=last_balance))
        if n == steps:
            break
        updated, fluxes = periodic_step(values, c, h, dt, numerical_flux)
        if not all(math.isfinite(v) for v in updated + fluxes):
            raise ArithmeticError('Non-finite result in ' + case['id'])
        last_balance = 0.0
        for i in range(count):
            mass_change = h * (updated[i] - values[i])
            net_inflow = dt * (fluxes[i] - fluxes[i+1])
            residual = mass_change - net_inflow
            max_balance = max(max_balance, abs(residual))
            last_balance = max(last_balance, abs(residual))
            if n in (0, steps-1):
                tables['balances'].append(dict(case=case['id'], step=n, i=i, width=h, dt=dt,
                                              old=values[i], new=updated[i], left_flux=fluxes[i],
                                              right_flux=fluxes[i+1], mass_change=mass_change,
                                              net_inflow=net_inflow, residual=residual))
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


def translated_averages(values, shift):
    """Integrate the translated piecewise-constant field over every fixed cell."""
    count = len(values)
    h = 1.0 / count
    result = []
    for i in range(count):
        left, right = i*h-shift, (i+1)*h-shift
        integral = 0.0
        for j in range(math.floor(left/h), math.ceil(right/h)):
            overlap = max(0.0, min(right, (j+1)*h)-max(left, j*h))
            integral += overlap * values[j % count]
        result.append(integral/h)
    return result


def algebra_examples(tables, profiles):
    old = [1.0, 2.0, 3.0, 2.0]
    for c in [1.0, -1.0, 0.0]:
        new, faces = periodic_step(old, c, 0.25, 0.125, upwind_flux)
        for i in range(4):
            tables['hand'].append(dict(c=c, i=i, old=old[i], left_flux=faces[i],
                                      right_flux=faces[i+1], new=new[i]))
    initial = [0.0, 1.0, 0.0, 0.0]
    for nu in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        c, dt = 2*nu, 0.125
        direct = translated_averages(initial, c*dt)
        # For abs(nu)>1 this call is deliberately outside the propagation gate.
        # It exposes the failure of the nearest-neighbor formula for long shifts.
        nearest, _ = periodic_step(initial, c, 0.25, dt, upwind_flux)
        for i in range(4):
            tables['remap'].append(dict(nu=nu, i=i, shift=c*dt, geometric=direct[i],
                                       nearest_upwind=nearest[i], difference=nearest[i]-direct[i]))
    for scenario, dt, steps in [('two_half_steps', 0.125, 2), ('one_full_step', 0.25, 1)]:
        values = initial[:]
        for n in range(steps+1):
            direct = translated_averages(initial, n*dt)
            for i in range(4):
                tables['reaverage'].append(dict(scenario=scenario, step=n, time=n*dt,
                                               i=i, numerical=values[i], exact=direct[i]))
            if n < steps:
                values, _ = periodic_step(values, 1.0, 0.25, dt, upwind_flux)
    case = dict(N=4, c=1.0, T=0.125, steps=1, profile='pulse')
    averages = exact_averages(case, profiles, 0.0)
    points = [float(profiles['pulse']['start'] <= (i+.5)/4 < profiles['pulse']['end']) for i in range(4)]
    for meaning, initial_values in [('cell_average', averages), ('center_value', points)]:
        updated, _ = periodic_step(initial_values, 1.0, 0.25, 0.125, upwind_flux)
        for i in range(4):
            tables['interpretations'].append(dict(meaning=meaning, i=i, initial=initial_values[i],
                                                   after_one_step=updated[i],
                                                   initial_total=0.25*math.fsum(initial_values),
                                                   final_total=0.25*math.fsum(updated)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/upwind.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate(config)
    tables = {name: [] for name in ['cells', 'history', 'balances', 'summary', 'convergence',
                                    'hand', 'remap', 'reaverage', 'interpretations']}
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
    algebra_examples(tables, config['profiles'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output, name, rows)
    print('rows: ' + ', '.join(name + '=' + str(len(rows)) for name, rows in tables.items()))
    print('max_mass_drift:', max(row['max_mass_drift'] for row in tables['summary']))
    print('max_balance_residual:', max(row['max_balance_residual'] for row in tables['summary']))


if __name__ == '__main__':
    main()
