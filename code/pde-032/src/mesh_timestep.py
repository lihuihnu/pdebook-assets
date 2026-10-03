"""Transparent heat solves, time-step trials and nonuniform Poisson FEM.

Only the standard library is needed. Reference formulas measure results;
they never replace a matrix solve or a time update.
"""
from pathlib import Path
import argparse
import bisect
import csv
import json
import math
import platform

HERE = Path(__file__).resolve().parents[1]


def factor_tridiagonal(lower, diagonal, upper):
    """One LU factorization; keep multipliers for repeated right-hand sides."""
    pivots = list(diagonal)
    multipliers = []
    for i in range(1, len(pivots)):
        m = lower[i-1] / pivots[i-1]
        multipliers.append(m)
        pivots[i] -= m * upper[i-1]
    if any(not math.isfinite(p) or p <= 0 for p in pivots):
        raise ArithmeticError("nonpositive or nonfinite tridiagonal pivot")
    return multipliers, pivots, list(upper)


def solve_factored(factors, rhs):
    multipliers, pivots, upper = factors
    y = list(rhs)
    for i in range(1, len(y)):
        y[i] -= multipliers[i-1] * y[i-1]
    u = [0.0] * len(y)
    u[-1] = y[-1] / pivots[-1]
    for i in range(len(y)-2, -1, -1):
        u[i] = (y[i] - upper[i] * u[i+1]) / pivots[i]
    if any(not math.isfinite(v) for v in u):
        raise ArithmeticError("nonfinite solution")
    return u


def heat_factors(n, rho):
    return factor_tridiagonal([-rho]*(n-1), [1+2*rho]*n, [-rho]*(n-1))


def backward_step(u, h, dt):
    return solve_factored(heat_factors(len(u), dt/(h*h)), u)


def trial_step(u, h, dt):
    coarse = backward_step(u, h, dt)
    half = backward_step(u, h, dt/2)
    fine = backward_step(half, h, dt/2)
    eta = 0.0
    for i in range(len(u)):
        eta = max(eta, abs(fine[i] - coarse[i]))
    return fine, eta, coarse, half


def error_norm(u, reference):
    return max(abs(a-b) for a, b in zip(u, reference))


def heat_references(N, T):
    h = 1/N
    lam = 4 * math.sin(math.pi*h/2)**2 / (h*h)
    shape = [math.sin(math.pi*i/N) for i in range(1, N)]
    exact = [math.exp(-math.pi**2*T)*v for v in shape]
    semidiscrete = [math.exp(-lam*T)*v for v in shape]
    return lam, shape, exact, semidiscrete


def uniform_heat(N, M, T, method):
    if N < 2 or M < 1 or not math.isfinite(T) or T <= 0 or method not in ["BE","FE"]:
        raise ValueError("valid grid, positive time and BE/FE method required")
    h, dt = 1/N, T/M
    rho = dt/(h*h)
    if method == "FE" and rho > .5 + 1e-14:
        raise ValueError("explicit heat stability limit exceeded")
    lam, u, exact, semidiscrete = heat_references(N, T)
    factors = heat_factors(N-1, rho) if method == "BE" else None
    for _ in range(M):
        old = u
        if method == "BE":
            u = solve_factored(factors, old)
        else:
            padded = [0.0] + old + [0.0]
            u = []
            for i in range(1, N):
                u.append(padded[i] + rho*(padded[i-1]-2*padded[i]+padded[i+1]))
    padded = [0.0] + u + [0.0]
    residual = 0.0
    for i in range(1, N):
        if method == "BE":
            value = (1+2*rho)*padded[i] - rho*(padded[i-1]+padded[i+1]) - old[i-1]
        else:
            previous = [0.0] + old + [0.0]
            value = padded[i] - previous[i] - rho*(previous[i-1]-2*previous[i]+previous[i+1])
        residual = max(residual, abs(value))
    summary = {"method": method, "N": N, "M": M, "h": h, "dt": dt,
        "lambda_h": lam, "error_inf": error_norm(u, exact),
        "space_error_inf": error_norm(semidiscrete, exact),
        "time_error_inf": error_norm(u, semidiscrete),
        "signed_mid_error": u[N//2-1]-exact[N//2-1],
        "last_update_residual_inf": residual, "work_units": (N-1)*M,
        "status": "complete"}
    return summary, u, old


def adaptive_heat(settings, T, rate):
    N = settings["N"]
    if N < 2 or not math.isfinite(rate) or rate <= 0 or T <= 0:
        raise ValueError("valid grid, positive finite rate and time required")
    h = 1/N
    lam, u, exact, semidiscrete = heat_references(N, T)
    dt, t, attempts, accepted, rejected, eta_sum = settings["initial_dt"], 0.0, [], 0, 0, 0.0
    samples = []
    first_reject = first_accept = False
    while t < T:
        if len(attempts) >= settings["max_trials"]:
            raise RuntimeError("adaptive trial budget exhausted")
        dt = min(dt, T-t)
        if dt < settings["min_dt"] or t+dt == t:
            raise RuntimeError("adaptive time step too small")
        start = list(u)
        fine, eta, coarse, half = trial_step(start, h, dt)
        threshold = rate*dt
        ok = eta <= threshold
        factor = settings["factor_max"] if eta == 0 else settings["safety"]*threshold/eta
        factor = min(settings["factor_max"], max(settings["factor_min"], factor))
        next_dt = dt*factor
        trial = len(attempts)
        is_last = ok and dt == T-t
        attempts.append({"rate": rate, "trial": trial, "t_start": t,
            "dt": dt, "t_end": t+dt if ok else t, "accepted": int(ok),
            "eta": eta, "threshold": threshold, "eta_rate": eta/dt,
            "factor": factor, "next_dt": next_dt,
            "start_mid": start[N//2-1], "coarse_mid": coarse[N//2-1],
            "half_mid": half[N//2-1], "fine_mid": fine[N//2-1]})
        take_sample = (not ok and not first_reject) or (ok and not first_accept) or is_last
        if take_sample:
            for i in range(1, N):
                samples.append({"rate": rate, "trial": trial, "i": i, "x": i/N,
                    "start": start[i-1], "coarse": coarse[i-1],
                    "half": half[i-1], "fine": fine[i-1]})
        if ok:
            first_accept = True
            u = fine
            t = T if is_last else t+dt
            accepted += 1
            eta_sum += eta
        else:
            first_reject = True
            rejected += 1
        dt = next_dt
    steps = [a["dt"] for a in attempts if a["accepted"]]
    summary = {"rate": rate, "N": N, "T": t, "accepted": accepted,
        "rejected": rejected, "linear_solves": 3*len(attempts),
        "work_units": 3*len(attempts)*(N-1), "first_accepted_dt": steps[0],
        "max_accepted_dt": max(steps), "last_accepted_dt": steps[-1],
        "eta_sum": eta_sum, "indicator_budget": rate*T,
        "space_error_inf": error_norm(semidiscrete, exact),
        "time_error_inf": error_norm(u, semidiscrete),
        "error_inf": error_norm(u, exact), "status": "complete"}
    return summary, attempts, samples, u


def poisson_fem(x):
    """Assemble K and integrate -12*x^2 against each linear basis exactly."""
    n = len(x)
    if n < 3 or x[0] != 0 or x[-1] != 1 or any(not math.isfinite(v) for v in x) or any(b <= a for a,b in zip(x,x[1:])):
        raise ValueError("increasing unit-interval mesh with an internal node required")
    diagonal, upper, load = [0.0]*n, [0.0]*(n-1), [0.0]*n
    element_data = []
    for k in range(n-1):
        a, b = x[k], x[k+1]
        length = b-a
        stiff = 1/length
        integral2, integral3 = (b**3-a**3)/3, (b**4-a**4)/4
        left = -12*(b*integral2-integral3)/length
        right = -12*(integral3-a*integral2)/length
        diagonal[k] += stiff
        diagonal[k+1] += stiff
        upper[k] = -stiff
        load[k] += left
        load[k+1] += right
        element_data.append((left, right))
    rhs = load[1:-1]
    rhs[0] -= upper[0]*0
    rhs[-1] -= upper[-1]*1
    factors = factor_tridiagonal(upper[1:-1], diagonal[1:-1], upper[1:-1])
    u = [0.0] + solve_factored(factors, rhs) + [1.0]
    residual = max(abs(diagonal[i]*u[i]+upper[i-1]*u[i-1]+upper[i]*u[i+1]-load[i])
                   for i in range(1,n-1))
    return u, element_data, residual


def interval_error(a, b, left, right):
    slope = (right-left)/(b-a)
    critical = min(b, max(a, (max(0.0,slope)/4)**(1/3)))
    values = [abs(left-a**4), abs(right-b**4),
              abs(left+slope*(critical-a)-critical**4)]
    return max(values), critical


def space_state(x):
    u, loads, residual = poisson_fem(x)
    elements = []
    for k in range(len(x)-1):
        a, b = x[k], x[k+1]
        eta = 1.5*(b-a)**2*b*b
        actual, critical = interval_error(a,b,u[k],u[k+1])
        elements.append({"k": k, "a": a, "b": b, "h": b-a, "eta": eta,
            "error_inf": actual, "critical_x": critical,
            "u_left": u[k], "u_right": u[k+1],
            "load_left": loads[k][0], "load_right": loads[k][1]})
    return u, elements, residual


def reconstruct(x, u, point):
    k = min(len(x)-2, max(0, bisect.bisect_right(x,point)-1))
    return u[k] + (u[k+1]-u[k])*(point-x[k])/(x[k+1]-x[k])


def spatial_experiments(settings):
    histories, elements_out, nodes, profiles, summaries = [], [], [], [], []
    def record(target, variant, iteration, x, final, selected=-1):
        u, elements, residual = space_state(x)
        prefix = {"target": target, "variant": variant, "iteration": iteration}
        for row in elements:
            elements_out.append({**prefix, **row})
        for i in range(len(x)):
            nodes.append({**prefix, "i": i, "x": x[i], "u": u[i],
                "exact": x[i]**4, "error": u[i]-x[i]**4})
        result = {**prefix, "cells": len(x)-1,
            "max_eta": max(e["eta"] for e in elements),
            "error_inf": max(e["error_inf"] for e in elements),
            "nodal_error_inf": max(abs(u[i]-x[i]**4) for i in range(len(x))),
            "residual_inf": residual, "selected_k": selected,
            "status": "complete" if final else "refine"}
        histories.append(result)
        if final:
            summaries.append(result)
            for j in range(settings["profile_points"]):
                point = j/(settings["profile_points"]-1)
                value = reconstruct(x,u,point)
                profiles.append({"target": target, "variant": variant,
                    "x": point, "u": value, "exact": point**4,
                    "error": value-point**4})
        return elements
    for target in settings["targets"]:
        x = [i/settings["initial_cells"] for i in range(settings["initial_cells"]+1)]
        iteration = 0
        while True:
            eta = [1.5*(b-a)**2*b*b for a,b in zip(x,x[1:])]
            k = max(range(len(eta)), key=lambda j:eta[j])
            final = eta[k] <= target
            record(target,"adaptive",iteration,x,final,-1 if final else k)
            if final:break
            if len(x)-1 >= settings["max_cells"]:raise RuntimeError("mesh budget exhausted")
            x.insert(k+1,(x[k]+x[k+1])/2)
            iteration += 1
        N = len(x)-1
        record(target,"uniform_same_count",0,[i/N for i in range(N+1)],True)
        # Conservative curvature bound max |u''| = 12 on the whole interval.
        count = max(2, math.ceil(math.sqrt(1.5/target)))
        record(target,"uniform_bound",0,[i/count for i in range(count+1)],True)
    return summaries, histories, elements_out, nodes, profiles


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k:format(v,".17g") if isinstance(v,float) else v for k,v in row.items()})


def run(config, output):
    settings = json.loads(config.read_text())
    heat, time, space = settings["heat"], settings["adaptive_time"], settings["adaptive_space"]
    if heat["T"] <= 0 or any(r <= 0 for r in time["rates"]) or any(e <= 0 for e in space["targets"]):
        raise ValueError("positive time and tolerances required")
    output.mkdir(parents=True,exist_ok=True)
    summaries, nodes, last_steps, values = [], [], [], {}
    cases = [(N,M,"BE","sweep") for N in heat["N"] for M in heat["M"]]
    for N in heat["joint_N"]:
        M = 3*N*N//5
        assert 5*M == 3*N*N
        for method in ["BE","FE"]:cases.append((N,M,method,"joint"))
    for N,M,method,group in cases:
        row, u, old = uniform_heat(N,M,heat["T"],method)
        case = f"{group}-{method}-N{N}-M{M}"
        row = {"case":case, "group":group, **row}
        summaries.append(row)
        lam, shape, exact, semi = heat_references(N,heat["T"])
        for i,v in enumerate([0.0]+u+[0.0]):
            nodes.append({"case":case, "i":i, "x":i/N, "u":v,
                "exact":0.0 if i in [0,N] else exact[i-1],
                "semidiscrete":0.0 if i in [0,N] else semi[i-1]})
        for i in range(1,N):
            last_steps.append({"case":case,"i":i,"old":old[i-1],"new":u[i-1]})
        if group == "sweep":values[(N,M)] = u
    differences = []
    for N,M in values:
        for direction,key in [("space",(2*N,M)),("time",(N,2*M))]:
            if key not in values:continue
            fine = values[key]
            restricted = fine[1::2] if direction == "space" else fine
            difference = error_norm(values[(N,M)],restricted)
            divisor = 3 if direction == "space" else 1
            next_key = (4*N,M) if direction == "space" else (N,4*M)
            order = ""
            if next_key in values:
                next_fine = values[next_key]
                restricted2 = next_fine[1::2] if direction == "space" else next_fine
                next_difference = error_norm(fine,restricted2)
                if difference > 1e-13 and next_difference > 1e-13:
                    order = math.log(difference/next_difference,2)
            differences.append({"direction":direction, "N":N,"M":M,
                "fine_N":key[0],"fine_M":key[1],"difference_inf":difference,
                "fine_error_estimate":difference/divisor,"observed_order":order})
    adaptive_summary, attempts, samples, adaptive_nodes = [], [], [], []
    for rate in time["rates"]:
        row, trials, sampled, u = adaptive_heat(time,heat["T"],rate)
        adaptive_summary.append(row);attempts.extend(trials);samples.extend(sampled)
        for i,v in enumerate([0.0]+u+[0.0]):
            adaptive_nodes.append({"rate":rate,"i":i,"x":i/time["N"],"u":v})
    spatial = spatial_experiments(space)
    datasets = {"heat_summary":summaries,"heat_nodes":nodes,"last_steps":last_steps,
        "refinement":differences,"time_summary":adaptive_summary,"time_trials":attempts,
        "time_samples":samples,"time_nodes":adaptive_nodes,
        "space_summary":spatial[0],"mesh_history":spatial[1],
        "mesh_elements":spatial[2],"mesh_nodes":spatial[3],"mesh_profiles":spatial[4]}
    for name, rows in datasets.items():write_csv(output/(name+".csv"),rows)
    eligible = [r for r in summaries if r["group"] == "sweep" and r["error_inf"] <= heat["target_error"]]
    best = min(eligible,key=lambda r:r["work_units"])
    log = [f"Python {platform.python_version()}; dimensionless T={heat['T']}.",
        "All heat states evolved by matrix solves or explicit updates; references only measure errors.",
        f"Uniform heat cases: {len(summaries)}; candidate sweeps: {len(values)}.",
        "work_units counts internal-unknown updates/solves, excludes factorization and wall-clock time.",
        f"Least work among candidates meeting {heat['target_error']}: {best['case']}, W={best['work_units']}, error={best['error_inf']:.17g}.",
        "Time indicators are asymptotic estimates; rate*T is an indicator budget, not a proved global error bound.",
        "Space indicator uses source-derived curvature; valid for this constant-coefficient exact-load 1D FEM, not a general residual estimator."]
    log += [f"{name}.csv: {len(rows)} rows" for name,rows in datasets.items()]
    (output/"run.txt").write_text("\n".join(log)+"\n")
    print("\n".join(log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=HERE/"config/problems.json")
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args = parser.parse_args()
    run(args.config,args.output)
