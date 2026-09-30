"""一维常速、固定端波动方程；求解只依赖 Python 标准库。"""
import argparse
import csv
import json
import math
import platform
from pathlib import Path


def first_step(displacement, velocity, dt, courant):
    """由位移与速度构造第一个时间层；端点固定为零。"""
    result = [0.0] * len(displacement)
    for i in range(1, len(displacement) - 1):
        curvature = displacement[i - 1] - 2.0 * displacement[i] + displacement[i + 1]
        result[i] = displacement[i] + dt * velocity[i] + 0.5 * courant**2 * curvature
    return result


def advance(previous, current, courant):
    """由相邻两层计算新层；不在旧列表中覆盖节点。"""
    result = [0.0] * len(current)
    for i in range(1, len(current) - 1):
        curvature = current[i - 1] - 2.0 * current[i] + current[i + 1]
        result[i] = 2.0 * current[i] - previous[i] + courant**2 * curvature
    return result


def layers(displacement, velocity, dt, courant, steps):
    previous = displacement[:]
    yield 0, previous
    current = first_step(previous, velocity, dt, courant)
    yield 1, current
    for n in range(1, steps):
        following = advance(previous, current, courant)
        previous, current = current, following
        yield n + 1, current


def energy(previous, current, h, dt, speed):
    """返回半时间层的守恒能量、两个展开项及非守恒的正项之和。"""
    kinetic = 0.0
    cross = 0.0
    midpoint_potential = 0.0
    for i in range(1, len(current) - 1):
        kinetic += 0.5 * h * ((current[i] - previous[i]) / dt)**2
    for i in range(len(current) - 1):
        old_gradient = (previous[i + 1] - previous[i]) / h
        new_gradient = (current[i + 1] - current[i]) / h
        cross += 0.5 * speed**2 * h * old_gradient * new_gradient
        midpoint_potential += 0.5 * speed**2 * h * ((old_gradient + new_gradient) / 2.0)**2
    return kinetic + cross, kinetic, cross, kinetic + midpoint_potential


def bump(x, center, width, power):
    y = (x - center) / width
    if abs(y) >= 1.0:
        return 0.0
    return (1.0 - y*y)**power


def bump_derivative(x, center, width, power):
    y = (x - center) / width
    if abs(y) >= 1.0:
        return 0.0
    return -2.0 * power * y * (1.0 - y*y)**(power - 1) / width


def sine_samples(cells, mode, amplitude):
    result = [0.0] * (cells + 1)
    for i in range(1, cells):
        result[i] = amplitude * math.sin(mode * math.pi * i / cells)
    return result


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def self_check():
    # 小网格、非零初速度；独立手算，而非调用另一份更新实现。
    initial = [0.0, 1.0, 1.0, 0.0]
    velocity = [0.0, 2.0, -1.0, 0.0]
    first = first_step(initial, velocity, 0.1, 0.5)
    require(first == [0.0, 1.075, 0.775, 0.0], 'first-step hand calculation')
    second = advance(initial, first, 0.5)
    require(max(abs(a-b) for a,b in zip(second, [0.0, 0.80625, 0.43125, 0.0])) < 1e-14,
            'second-step hand calculation')
    require(initial == [0.0, 1.0, 1.0, 0.0], 'input was overwritten')
    require(velocity == [0.0, 2.0, -1.0, 0.0], 'velocity was overwritten')
    require(first == [0.0, 1.075, 0.775, 0.0], 'old layer was overwritten')
    require(first_step([0.0]*4, [0.0]*4, 0.1, 0.8) == [0.0]*4, 'zero solution')
    largest_defect = 0.0
    largest_energy_drift = 0.0
    cases = 0
    # 正弦闭式仅作参考；实际节点始终由三点更新产生。
    for cells in (2, 3, 10):
        h = 1.0 / cells
        for courant in (0.25, 0.8, 1.0):
            dt = courant * h
            for mode in range(1, cells):
                phi = 2.0 * math.asin(courant * math.sin(mode * math.pi / (2*cells)))
                initial = sine_samples(cells, mode, 0.7)
                velocity = sine_samples(cells, mode, -0.4)
                old = None
                initial_energy = None
                for n, values in layers(initial, velocity, dt, courant, 30):
                    coefficient = 0.7 * math.cos(n*phi) - 0.4*dt/math.sin(phi)*math.sin(n*phi)
                    expected = sine_samples(cells, mode, coefficient)
                    defect = max(abs(a-b) for a,b in zip(values, expected))
                    largest_defect = max(largest_defect, defect)
                    require(defect < 2e-12, 'discrete modal reference')
                    require(values[0] == values[-1] == 0.0, 'fixed ends')
                    if old is not None:
                        conserved, kinetic, cross, positive = energy(old, values, h, dt, 1.0)
                        if initial_energy is None:
                            initial_energy = conserved
                        drift = abs(conserved - initial_energy) / initial_energy
                        largest_energy_drift = max(largest_energy_drift, drift)
                        require(drift < 2e-11, 'discrete energy drift')
                        lower = (1.0 - courant**2) * kinetic + positive - kinetic
                        require(conserved >= lower - 1e-11, 'energy lower bound')
                    old = values
                cases += 1
    # 离散分部求和：用不同节点向量检查符号和端点边项。
    a = [0.0, 0.3, -0.8, 0.2, 0.0]
    b = [0.0, -0.5, 0.7, 0.9, 0.0]
    h = 0.25
    lhs = sum(h*(a[i-1]-2*a[i]+a[i+1])/h**2*b[i] for i in range(1, 4))
    rhs = -sum((a[i+1]-a[i])*(b[i+1]-b[i])/h for i in range(4))
    require(abs(lhs-rhs) < 1e-14, 'summation by parts')
    return cases, largest_defect, largest_energy_drift


def write_csv(path, rows):
    require(bool(rows), 'empty output ' + str(path))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output):
    speed = config['wave_speed']
    require(math.isfinite(speed) and speed > 0.0, 'wave_speed must be positive')
    pulse = config['pulse']
    center, width, power = pulse['center'], pulse['half_width'], pulse['power']
    require(isinstance(power, int) and power >= 6, 'pulse power must be an integer >= 6')
    require(0.0 < center-width < center+width < 1.0, 'pulse must lie inside (0,1)')
    require(speed * pulse['final_time'] < 2.0-center-width, 'reference excludes a left-end reflection')
    require(0.0 <= min(pulse['snapshot_times']) <= max(pulse['snapshot_times']) <= pulse['final_time'],
            'snapshot time outside interval')
    conv = config['convergence']
    require(isinstance(conv['steps_per_cell'], int) and conv['steps_per_cell'] > 0, 'steps_per_cell')
    specs = [('pulse', pulse['cells'], pulse['steps'], pulse['final_time'])]
    for cells in conv['cells']:
        specs.append(('sine', cells, cells*conv['steps_per_cell'], conv['final_time']))
    snapshots, finals, histories, summaries = [], [], [], []
    for case, cells, steps, final_time in specs:
        require(isinstance(cells, int) and cells >= 2, 'cells must be an integer >= 2')
        require(isinstance(steps, int) and steps >= 1, 'steps must be a positive integer')
        require(math.isfinite(final_time) and final_time > 0, 'final_time must be positive')
        h, dt = 1.0/cells, final_time/steps
        courant = speed*dt/h
        require(0.0 < courant <= 1.0, 'use a Courant number in (0,1]')
        selected = set()
        if case == 'pulse':
            for t in pulse['snapshot_times']:
                n = round(t/dt)
                require(abs(n*dt-t) < 1e-12, 'snapshot must be an actual time layer')
                selected.add(n)
            initial, velocity = [], []
            for i in range(cells+1):
                x = i*h
                initial.append(bump(x, center, width, power))
                velocity.append(-speed*bump_derivative(x, center, width, power))
            def exact(x, t):
                return bump(x-speed*t, center, width, power) - bump(2.0-x-speed*t, center, width, power)
        else:
            amp, vel = conv['displacement_amplitude'], conv['velocity_amplitude']
            require(math.isfinite(amp) and math.isfinite(vel), 'finite sine amplitudes required')
            initial = sine_samples(cells, 1, amp)
            velocity = sine_samples(cells, 1, vel)
            def exact(x, t):
                return (amp*math.cos(speed*math.pi*t) + vel/(speed*math.pi)*math.sin(speed*math.pi*t))*math.sin(math.pi*x)
        old = None
        initial_energy = None
        max_error = max_drift = max_residual = 0.0
        naive_min = math.inf
        naive_max = -math.inf
        for n, values in layers(initial, velocity, dt, courant, steps):
            require(all(math.isfinite(v) for v in values), 'non-finite numerical state')
            require(values[0] == values[-1] == 0.0, 'nonzero boundary')
            error = 0.0
            for i in range(cells+1):
                x = i*h
                reference = exact(x, n*dt)
                if i in (0, cells):
                    require(abs(reference) < 1e-13, 'reference boundary')
                    reference = 0.0
                difference = values[i] - reference
                error = max(error, abs(difference))
                row = dict(case=case, cells=cells, n=n, time=n*dt, i=i, x=x,
                           numerical=values[i], exact=reference, error=difference)
                if case == 'pulse' and n in selected:
                    snapshots.append(row)
                if n == steps:
                    finals.append(row)
            max_error = max(max_error, error)
            if old is not None:
                conserved, kinetic, cross, naive = energy(old, values, h, dt, speed)
                if initial_energy is None:
                    initial_energy = conserved
                    require(initial_energy > 0.0, 'positive initial discrete energy required')
                drift = (conserved-initial_energy)/initial_energy
                max_drift = max(max_drift, abs(drift))
                naive_min, naive_max = min(naive_min, naive), max(naive_max, naive)
                histories.append(dict(case=case, cells=cells, n=n, time=n*dt,
                                      half_time=(n-0.5)*dt, energy=conserved, kinetic=kinetic,
                                      cross_potential=cross, positive_energy=naive,
                                      relative_energy_drift=drift, max_error=error))
                if n >= 2:
                    for i in range(1, cells):
                        terms = [values[i], -2.0*old[i], older[i],
                                 -courant**2*(old[i-1]-2.0*old[i]+old[i+1])]
                        scaled = abs(sum(terms))/max(1.0, sum(abs(a) for a in terms))
                        max_residual = max(max_residual, scaled)
            older, old = old, values
        require(max_drift < 2e-11, 'energy conservation check failed')
        require(max_residual < 1e-13, 'recurrence residual check failed')
        summaries.append(dict(case=case, cells=cells, steps=steps, h=h, dt=dt, courant=courant,
                              final_time=final_time, final_max_error=error, trajectory_max_error=max_error,
                              initial_energy=initial_energy, max_relative_energy_drift=max_drift,
                              positive_energy_range_relative=(naive_max-naive_min)/initial_energy,
                              max_scaled_residual=max_residual))
    convergence = []
    previous_error = None
    for row in summaries:
        if row['case'] != 'sine':
            continue
        error = row['final_max_error']
        convergence.append(dict(cells=row['cells'], steps=row['steps'], h=row['h'], dt=row['dt'],
                                final_time=row['final_time'], max_error=error,
                                ratio='' if previous_error is None else previous_error/error))
        previous_error = error
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('snapshots', snapshots), ('final_nodes', finals), ('history', histories),
                       ('summary', summaries), ('convergence', convergence)]:
        write_csv(output/(name+'.csv'), rows)
    return summaries, (len(snapshots), len(finals), len(histories))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root/'config/wave.json')
    parser.add_argument('--output', type=Path, default=root/'results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    cases, modal_defect, energy_drift = self_check()
    summaries, counts = run(config, args.output)
    lines = [f'Python {platform.python_version()}', 'Solver dependencies: standard library only',
             'Command: python src/wave_centered.py (from companion directory)',
             'Boundary: homogeneous Dirichlet on [0,1]',
             'First layer: displacement + dt*velocity + 0.5*courant^2*second_difference',
             f'Self-check: hand calculation, zero state, unchanged inputs, summation by parts, {cases} modal trajectories',
             f'Max modal-reference defect: {modal_defect:.12e}',
             f'Max self-check relative energy drift: {energy_drift:.12e}',
             f'Saved rows: snapshots={counts[0]}, final_nodes={counts[1]}, history={counts[2]}',
             'All time layers checked; only configured pulse snapshots and terminal nodal fields saved.',
             'History energy is at half_time; max_error is at time.']
    for row in summaries:
        lines.append(f"{row['case']} N={row['cells']} M={row['steps']} nu={row['courant']:.8g} "
                     f"E_final={row['final_max_error']:.12e} "
                     f"energy_drift={row['max_relative_energy_drift']:.12e}")
    lines.append('PASS: finite states, exact boundary assignment, energy and scaled recurrence residual checks')
    report = '\n'.join(lines) + '\n'
    (args.output/'run.txt').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
