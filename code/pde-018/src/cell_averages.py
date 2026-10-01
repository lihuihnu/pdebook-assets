#!/usr/bin/env python3
"""Exact cell integrals for periodic advection; no PDE time-stepping solver."""

import argparse
import csv
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def cell_averages(edges, primitive, shift):
    averages = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        mass = primitive(right - shift) - primitive(left - shift)
        averages.append(mass / (right - left))
    return averages


def profile_functions(name, config):
    """Return a point value, a continuous integral primitive, and period mass."""
    amplitude = config['cosine_amplitude']
    a, b = config['pulse_interval']
    if name == 'cosine':
        def value(x):
            return 1 + amplitude * math.cos(2 * math.pi * x)

        def primitive(x):
            return x + amplitude * math.sin(2 * math.pi * x) / (2 * math.pi)

        return value, primitive, 1.0
    if name == 'pulse':
        def value(x):
            fraction = x - math.floor(x)
            return 1.0 if a <= fraction < b else 0.0

        def primitive(x):
            periods = math.floor(x)
            fraction = x - periods
            return (b - a) * periods + max(0.0, min(fraction - a, b - a))

        return value, primitive, b - a
    raise ValueError('Unknown profile: ' + name)


def save_csv(output, name, rows):
    with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    if config['domain'] != [0, 1]:
        raise ValueError('This example uses a unit periodic interval.')
    a, b = config['pulse_interval']
    if not 0 <= a < b <= 1:
        raise ValueError('Pulse endpoints must satisfy 0 <= a < b <= 1.')
    times = config['times']
    if times[0] != 0 or any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
        raise ValueError('Times must start at zero and increase strictly.')
    for edges in config['grids'].values():
        if edges[0] != 0 or edges[-1] != 1 or any(r <= l for l, r in zip(edges, edges[1:])):
            raise ValueError('Grid must partition [0, 1] into positive-width cells.')

    cells, totals, faces, balances, comparison = [], [], [], [], []
    for name in config['profiles']:
        value, primitive, expected_total = profile_functions(name, config)
        for grid, edges in config['grids'].items():
            for c in config['velocities']:
                for t in times:
                    averages = cell_averages(edges, primitive, c * t)
                    masses, sampled_masses = [], []
                    for i, average in enumerate(averages):
                        left, right = edges[i], edges[i + 1]
                        width = right - left
                        center = (left + right) / 2
                        center_value = value(center - c * t)
                        mass = width * average
                        masses.append(mass)
                        sampled_masses.append(width * center_value)
                        cells.append(dict(profile=name, grid=grid, c=c, time=t, i=i,
                                          left=left, right=right, center=center, width=width,
                                          average=average, center_value=center_value, mass=mass))
                    total = math.fsum(masses)
                    totals.append(dict(profile=name, grid=grid, c=c, time=t,
                                       total=total, expected_total=expected_total,
                                       total_error=total-expected_total,
                                       midpoint_total=math.fsum(sampled_masses)))
                for ta, tb in zip(times, times[1:]):
                    before = cell_averages(edges, primitive, c * ta)
                    after = cell_averages(edges, primitive, c * tb)
                    transports = []
                    for j, x in enumerate(edges):
                        transport = primitive(x - c * ta) - primitive(x - c * tb)
                        transports.append(transport)
                        faces.append(dict(profile=name, grid=grid, c=c, ta=ta, tb=tb,
                                          j=j, x=x, transport=transport))
                    for i, (left, right) in enumerate(zip(edges, edges[1:])):
                        change = (right - left) * (after[i] - before[i])
                        net = transports[i] - transports[i + 1]
                        balances.append(dict(profile=name, grid=grid, c=c, ta=ta, tb=tb,
                                             i=i, mass_change=change, inflow_minus_outflow=net,
                                             residual=change-net))

    value, primitive, _ = profile_functions('cosine', config)
    previous_gap = None
    for n in config['comparison']['N']:
        edges = [i / n for i in range(n + 1)]
        shift = config['comparison']['c'] * config['comparison']['time']
        averages = cell_averages(edges, primitive, shift)
        gap = max(abs(averages[i] - value((edges[i]+edges[i+1])/2-shift)) for i in range(n))
        comparison.append(dict(N=n, h=1/n, max_gap=gap,
                               previous_ratio='' if previous_gap is None else previous_gap/gap))
        previous_gap = gap

    plot = config['plot']
    curve = []
    value, _, _ = profile_functions('cosine', config)
    for j in range(plot['cosine_points']):
        x = j / (plot['cosine_points'] - 1)
        curve.append(dict(profile='cosine', x=x, value=value(x-plot['c']*plot['time'])))
    # Exact polygonal graph, including both sides of each jump; not sampled ramps.
    left, right = a + plot['c'] * plot['time'], b + plot['c'] * plot['time']
    if not 0 < left < right < 1:
        raise ValueError('The selected pulse plot must not cross the periodic boundary.')
    for x, y in [(0, 0), (left, 0), (left, 1), (right, 1), (right, 0), (1, 0)]:
        curve.append(dict(profile='pulse', x=x, value=y))

    output.mkdir(parents=True, exist_ok=True)
    for name, records in [('cells', cells), ('totals', totals), ('faces', faces),
                          ('balances', balances), ('comparison', comparison), ('exact_curve', curve)]:
        save_csv(output, name, records)
    print(f'{len(totals)} snapshots; {len(cells)} cells; {len(faces)} face integrals; {len(balances)} balances')
    print(f'Max total error: {max(abs(r["total_error"]) for r in totals):.3e}')
    print(f'Max space-time balance residual: {max(abs(r["residual"]) for r in balances):.3e}')
    for row in comparison:
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'config/averages.json')
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
