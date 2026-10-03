"""Known-solution verification and explicitly marked fault diagnostics.

Core solver and CSV generation use only the Python standard library.
Run: python experiments/pde-031/src/verify_poisson.py
"""
from pathlib import Path
import csv
import json
import math

BASE = Path(__file__).resolve().parents[1]


def four_cell_check():
    diagonal = [2.0, 2.0, 2.0]
    rhs = [65 / 64, 1 / 16, 129 / 64]
    for i in range(1, 3):
        multiplier = -1 / diagonal[i - 1]
        diagonal[i] += multiplier
        rhs[i] -= multiplier * rhs[i - 1]
    solution = [0.0, 0.0, 0.0]
    solution[2] = rhs[2] / diagonal[2]
    for i in range(1, -1, -1):
        solution[i] = (rhs[i] + solution[i + 1]) / diagonal[i]
    return solution


def thomas(rhs):
    """Solve the fixed Poisson tridiagonal system by elimination/back-substitution."""
    n = len(rhs)
    if n == 0:
        raise ValueError("at least one internal node is required")
    diagonal = [2.0] * n
    load = list(rhs)
    for i in range(1, n):
        multiplier = -1 / diagonal[i - 1]
        diagonal[i] += multiplier
        load[i] -= multiplier * load[i - 1]
    u = [0.0] * n
    u[-1] = load[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        u[i] = (load[i] + u[i + 1]) / diagonal[i]
    return u


def exact_and_source(model, x):
    if model == "sine":
        return math.sin(math.pi * x), math.pi ** 2 * math.sin(math.pi * x)
    if model == "quadratic":
        return 1 + x + x * (1 - x), 2.0
    if model == "quartic":
        return 1 + x + x ** 2 * (1 - x) ** 2, -2 + 12 * x - 12 * x ** 2
    raise ValueError("unknown model")


def discrete_reference(model, N, i):
    """Closed forms derived in the manuscript; not used by the solver."""
    x, h = i / N, 1 / N
    value, _ = exact_and_source(model, x)
    if model == "sine":
        eigenvalue = 4 * math.sin(math.pi * h / 2) ** 2 / h ** 2
        return math.pi ** 2 / eigenvalue * value
    if model == "quartic":
        return value + h ** 2 * x * (1 - x)
    return value


def residual(rhs, internal):
    zero_extended = [0.0] + list(internal) + [0.0]
    return [rhs[i] - 2 * zero_extended[i + 1] + zero_extended[i] + zero_extended[i + 2]
            for i in range(len(internal))]


def norm_inf(vector):
    return max(map(abs, vector), default=0.0)


def writerows(path, rows):
    if not rows:
        raise ValueError("empty data table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                             for key, value in row.items()})


def run():
    cfg = json.loads((BASE / "config/problems.json").read_text())
    results = BASE / "results"
    results.mkdir(exist_ok=True)
    tables = {name: [] for name in ["loads", "history", "iterates", "nodes", "fluxes",
                                   "balances", "summary", "refinement"]}
    solutions = {}
    for model in cfg["models"]:
        variants = ["correct"] + (cfg["quartic_diagnostics"] if model == "quartic" else [])
        for variant in variants:
            for N in cfg["meshes"]:
                h = 1 / N
                case = dict(model=model, variant=variant, N=N)
                left, right = (0.0, 0.0) if model == "sine" else (1.0, 2.0)
                used_right = 0.0 if variant == "omit_right" else right
                coords = [i * h for i in range(1, N)]
                target = [exact_and_source(model, x)[0] for x in coords]
                source = [exact_and_source(model, x)[1] for x in coords]
                used_source = [f + cfg["source_bias"] if variant == "biased_source" else f
                               for f in source]
                rhs = [h ** 2 * f for f in source]
                rhs[0] += left
                rhs[-1] += right
                used_rhs = [h ** 2 * f for f in used_source]
                used_rhs[0] += left
                used_rhs[-1] += used_right
                reference = [discrete_reference(model, N, i) for i in range(1, N)]
                for i, x in enumerate(coords):
                    tables["loads"].append(dict(**case, i=i + 1, x=x, exact=target[i],
                        discrete_exact=reference[i], f_target=source[i], f_used=used_source[i],
                        b_target=rhs[i], b_used=used_rhs[i], left_target=left,
                        right_target=right, left_used=left, right_used=used_right))
                if variant == "fixed_jacobi":
                    internal = [left + (right - left) * x for x in coords]
                    snapshots = [internal.copy()]
                    for _ in range(cfg["jacobi_sweeps"]):
                        v = [0.0] + internal + [0.0]
                        internal = [(used_rhs[i] + v[i] + v[i + 2]) / 2
                                    for i in range(N - 1)]
                        snapshots.append(internal.copy())
                    status = "iteration_budget"
                else:
                    internal = thomas(used_rhs)
                    if variant == "balanced_perturbation":
                        internal = [u + cfg["perturbation_amplitude"] * math.sin(2 * math.pi * x)
                                    for u, x in zip(internal, coords)]
                        status = "perturbed_after_solve"
                    else:
                        status = "direct_solve"
                    snapshots = [internal.copy()]
                for k, v in enumerate(snapshots):
                    used_res, target_res = residual(used_rhs, v), residual(rhs, v)
                    algebraic = norm_inf([u - ref for u, ref in zip(v, reference)])
                    error = norm_inf([u - true for u, true in zip(v, target)])
                    tables["history"].append(dict(**case, k=k, assembled_residual_inf=norm_inf(used_res),
                        target_residual_inf=norm_inf(target_res), target_pde_residual_inf=norm_inf(target_res)/h**2,
                        target_algebraic_error_inf=algebraic, error_inf=error,
                        status=status if k == len(snapshots) - 1 else "iteration_running"))
                    for i, u in enumerate(v, 1):
                        tables["iterates"].append(dict(**case, k=k, i=i, u=u))
                full = [left] + internal + [right]
                assembled_full = [left] + internal + [used_right]
                exact_full = [left] + target + [right]
                for i, u in enumerate(full):
                    tables["nodes"].append(dict(**case, i=i, x=i*h, u=u,
                        u_assembled=assembled_full[i], exact=exact_full[i], error=u-exact_full[i]))
                q = [-(full[i + 1] - full[i]) / h for i in range(N)]
                used_q = [-(assembled_full[i + 1] - assembled_full[i]) / h for i in range(N)]
                for i in range(N):
                    tables["fluxes"].append(dict(**case, face=i, x=(i + 0.5)*h, q=q[i], q_used=used_q[i]))
                local = [(q[i + 1] - q[i]) / h - source[i] for i in range(N - 1)]
                local_used = [(used_q[i + 1] - used_q[i]) / h - used_source[i] for i in range(N - 1)]
                for i in range(N - 1):
                    tables["balances"].append(dict(**case, i=i + 1, x=coords[i],
                        local_pde_defect=local[i], local_used_pde_defect=local_used[i],
                        control_volume_defect=h*local[i]))
                balance = q[-1] - q[0] - h * math.fsum(source)
                used_balance = used_q[-1] - used_q[0] - h * math.fsum(used_source)
                error = [u - true for u, true in zip(internal, target)]
                summary = dict(**case, h=h, status=status, updates=len(snapshots)-1,
                    error_inf=norm_inf(error), error_l2h=math.sqrt(h*math.fsum(e*e for e in error)),
                    target_algebraic_error_inf=norm_inf([u-r for u,r in zip(internal,reference)]),
                    discretization_error_inf=norm_inf([r-t for r,t in zip(reference,target)]),
                    assembled_residual_inf=norm_inf(residual(used_rhs,internal)),
                    target_residual_inf=norm_inf(residual(rhs,internal)),
                    target_pde_residual_inf=norm_inf(residual(rhs,internal))/h**2,
                    max_local_pde_defect=norm_inf(local), balance_target=balance, balance_used=used_balance,
                    boundary_output_error_inf=0.0, boundary_assembly_error_inf=abs(used_right-right))
                tables["summary"].append(summary)
                solutions[model, variant, N] = (full, summary)
            meshes = cfg["meshes"]
            differences = []
            for coarse, fine in zip(meshes, meshes[1:]):
                uc, sc = solutions[model, variant, coarse]
                uf, sf = solutions[model, variant, fine]
                differences.append(norm_inf([a-b for a,b in zip(uc,uf[::2])]))
            for j, (coarse, fine) in enumerate(zip(meshes, meshes[1:])):
                sc, sf = solutions[model,variant,coarse][1], solutions[model,variant,fine][1]
                meaningful = min(sc["error_inf"],sf["error_inf"]) > cfg["order_error_floor"]
                d = differences[j]
                next_d = differences[j+1] if j+1 < len(differences) else None
                dmeaningful = next_d is not None and min(d,next_d) > cfg["order_error_floor"]
                tables["refinement"].append(dict(model=model,variant=variant,N=coarse,N_fine=fine,
                    error_coarse=sc["error_inf"],error_fine=sf["error_inf"],difference_inf=d,
                    p_error=math.log2(sc["error_inf"]/sf["error_inf"]) if meaningful else "",
                    p_difference=math.log2(d/next_d) if dmeaningful else "",
                    error_order_status="computed" if meaningful else "below_error_floor",
                    difference_order_status="computed" if dmeaningful else "missing_third_grid" if next_d is None else "below_error_floor"))
    for name, data in tables.items():
        writerows(results / (name + ".csv"), data)
    log = ["pde-031 known-solution verification; all quantities dimensionless",
           "Residual convention b-Au; all deliberate diagnostics are labeled.",
           "Fixed Jacobi reports iteration_budget, never inferred convergence.",
           "Norm l2h is sqrt(h*sum(interior error squared)), not continuous reconstruction error."]
    log += [f"{name}: {len(data)} rows" for name,data in tables.items()]
    log += [f"cases: {len(tables['summary'])}", "four-cell solution: " + ", ".join(format(v,'.17g') for v in four_cell_check())]
    (results / "run.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    run()
