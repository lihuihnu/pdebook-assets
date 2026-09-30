"""E013-01: FTCS stability with a specified small sine perturbation.

All solution values are computed by node updates. Modal formulas are used only
as independent checks and recorded references. Solver: Python standard library.
"""
import csv
import json
import math
import platform
from pathlib import Path


def advance(values, ratio):
    """由当前层计算下一层；两个端点维持零边界。"""
    next_values = [0.0] * len(values)
    for i in range(1, len(values) - 1):
        next_values[i] = values[i] + ratio * (
            values[i - 1] - 2.0 * values[i] + values[i + 1]
        )
    return next_values


def amplification(mode, intervals, ratio):
    return 1.0 - 4.0 * ratio * math.sin(mode * math.pi / (2 * intervals))**2


def exact_value(x, t, alpha, epsilon, mode):
    low = math.exp(-alpha * math.pi**2 * t) * math.sin(math.pi * x)
    high = epsilon * math.exp(-alpha * (mode * math.pi)**2 * t)
    return low + high * math.sin(mode * math.pi * x)


def check_update():
    old = [0.0, 1.0, 2.0, 1.0, 0.0]
    assert advance(old, 0.25) == [0.0, 1.0, 1.5, 1.0, 0.0]
    assert old == [0.0, 1.0, 2.0, 1.0, 0.0]
    # N=2: the exact fixed-grid modal boundary is r=1, not r=1/2.
    assert advance([0.0, 1.0, 0.0], 0.5) == [0.0, 0.0, 0.0]
    assert advance([0.0, 1.0, 0.0], 1.0) == [0.0, -1.0, 0.0]
    assert abs(advance([0.0, 1.0, 0.0], 1.1)[1]) > 1.0
    # Check every mode against the stencil, without diagonalizing a matrix.
    for ratio in (0.4, 0.5, 0.6):
        for mode in range(1, 20):
            values = [0.0] * 21
            for i in range(1, 20):
                values[i] = math.sin(mode * math.pi * i / 20)
            updated = advance(values, ratio)
            factor = amplification(mode, 20, ratio)
            assert max(abs(updated[i] - factor * values[i]) for i in range(21)) < 2e-14


def simulate(config, case):
    intervals = config['intervals']
    steps = case['steps']
    mode = config['perturbation_mode']
    if type(intervals) is not int or intervals < 2:
        raise ValueError('intervals must be an integer >= 2')
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if type(mode) is not int or not 1 <= mode < intervals:
        raise ValueError('perturbation mode must be between 1 and N-1')
    for name in ('alpha', 'final_time', 'epsilon'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + ' must be positive and finite')
    alpha, epsilon = config['alpha'], config['epsilon']
    h = 1.0 / intervals
    dt = config['final_time'] / steps
    ratio = alpha * dt / h**2
    if not math.isfinite(ratio):
        raise ValueError('non-finite ratio')
    # Both sides of the stability boundary are intentionally part of this study.
    if not math.isclose(ratio, case['expected_ratio'], rel_tol=1e-12):
        raise ValueError('steps, final_time and expected_ratio disagree')
    low_factor = amplification(1, intervals, ratio)
    high_factor = amplification(mode, intervals, ratio)
    values = [0.0] * (intervals + 1)
    for i in range(1, intervals):
        values[i] = exact_value(i * h, 0.0, alpha, epsilon, mode)
    nodes, history = [], []
    for n in range(steps + 1):
        t = n * dt
        error_inf = modal_check_inf = 0.0
        for i, value in enumerate(values):
            x = i * h
            exact = discrete_reference = 0.0
            if 0 < i < intervals:
                exact = exact_value(x, t, alpha, epsilon, mode)
                discrete_reference = low_factor**n * math.sin(math.pi * x)
                discrete_reference += epsilon * high_factor**n * math.sin(mode * math.pi * x)
            error_inf = max(error_inf, abs(value - exact))
            modal_check_inf = max(modal_check_inf, abs(value - discrete_reference))
            nodes.append([case['id'], n, t, i, x, value, exact, value - exact])
        history.append([case['id'], n, t, dt, ratio, error_inf,
                        min(values), max(values), modal_check_inf])
        assert values[0] == values[-1] == 0.0
        scale = max(1.0, abs(low_factor)**n, epsilon * abs(high_factor)**n)
        assert modal_check_inf < 2e-8 * scale
        if n == steps:
            break
        next_values = advance(values, ratio)
        if not all(math.isfinite(v) for v in next_values):
            raise ArithmeticError('non-finite temperature; result not clipped')
        if ratio <= 0.5:
            assert min(next_values) >= -1e-14
            assert max(next_values) <= max(values) + 1e-14
        values = next_values
    return nodes, history


def write_csv(path, header, rows):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'config/stability.json').read_text())
    check_update()
    nodes, history, modes, snapshots = [], [], [], []
    log = [f'Python {platform.python_version()}; standard-library solver',
           f'N={config["intervals"]}, alpha={config["alpha"]}, epsilon={config["epsilon"]}, mode={config["perturbation_mode"]}, T={config["final_time"]}']
    for case in config['cases']:
        case_nodes, case_history = simulate(config, case)
        nodes.extend(case_nodes)
        history.extend(case_history)
        ratio = case_history[0][4]
        for mode in range(1, config['intervals']):
            modes.append([case['id'], mode, ratio,
                          amplification(mode, config['intervals'], ratio)])
        for time in config['snapshot_times']:
            n = round(time / case_history[0][3])
            row = case_history[n]
            if not math.isclose(row[2], time, abs_tol=1e-14):
                raise ValueError('snapshot time is not a time layer')
            snapshots.append(row)
            log.append(f'{case["id"]}: n={n:2d}, t={time:.3f}, error_inf={row[5]:.9e}, min={row[6]:.9e}, max={row[7]:.9e}')
        log.append(f'{case["id"]}: max modal-check discrepancy={max(q[8] for q in case_history):.9e}')
    output = root / 'results'
    output.mkdir(exist_ok=True)
    header = ['case', 'n', 't', 'dt', 'r', 'error_inf', 'minimum', 'maximum', 'modal_check_inf']
    write_csv(output / 'nodes.csv', ['case', 'n', 't', 'i', 'x', 'numerical', 'exact', 'error'], nodes)
    write_csv(output / 'history.csv', header, history)
    write_csv(output / 'snapshots.csv', header, snapshots)
    write_csv(output / 'modes.csv', ['case', 'mode', 'r', 'factor'], modes)
    log.append('PASS: hand update, old-layer preservation, N=2 boundary, all 19 sine modes, full trajectory closed form, finite values, zero boundaries, stable-case maximum')
    (output / 'run.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    main()
