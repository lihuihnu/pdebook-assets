"""Teaching Newton solver: scalar example, tridiagonal steps and halving search.

Core uses only the Python standard library. Exact nodes are runner diagnostics,
never inputs to the Newton step, line search or stopping criterion.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def one_node_newton():
    z = 0.0
    for k in range(40):
        residual = 2.0 * z + 250.0 * z ** 3 - 252.0
        if abs(residual) <= max(1e-13, 1e-10 * 252.0):
            return z
        z -= residual / (2.0 + 750.0 * z ** 2)
    raise RuntimeError("Newton iteration reached its limit")


def norm(values):
    return math.hypot(*values)


def residual(u, b, coefficient):
    n = len(u)
    return [(2.0 * u[i] - (u[i - 1] if i else 0.0)
             - (u[i + 1] if i + 1 < n else 0.0)
             + coefficient * u[i] ** 3 - b[i]) for i in range(n)]


def jacobian_diagonal(u, coefficient):
    return [2.0 + 3.0 * coefficient * value ** 2 for value in u]


def tridiagonal_solve(diagonal, rhs):
    """Solve SPD tridiagonal system with both off-diagonals equal to -1."""
    d, r = diagonal.copy(), rhs.copy()
    if len(d) != len(r) or not d:
        raise ValueError("nonempty matching diagonal and right hand side required")
    for i in range(len(d)):
        if i:
            multiplier = -1.0 / d[i - 1]
            d[i] -= multiplier * -1.0
            r[i] -= multiplier * r[i - 1]
        if not math.isfinite(d[i]) or d[i] <= 0.0:
            raise ArithmeticError("nonpositive or nonfinite elimination pivot")
        if not math.isfinite(r[i]):
            raise ArithmeticError("nonfinite eliminated right hand side")
    result = [0.0] * len(d)
    result[-1] = r[-1] / d[-1]
    for i in range(len(d) - 2, -1, -1):
        result[i] = (r[i] + result[i + 1]) / d[i]
    return result


def linear_defect(diagonal, delta, f):
    n = len(delta)
    return [diagonal[i] * delta[i] - (delta[i - 1] if i else 0.0)
            - (delta[i + 1] if i + 1 < n else 0.0) + f[i] for i in range(n)]


def newton(b, coefficient, initial, method, rtol=1e-10, atol=1e-13,
           step_tol=1e-14, maxiter=80, sigma=1e-4, max_backtracks=20):
    """Return every actual iterate, direction, trial and explicit stop status."""
    if method not in ("full", "damped"):
        raise ValueError("method must be full or damped")
    if not b or len(initial) != len(b):
        raise ValueError("nonempty matching initial vector and load required")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("nonnegative finite reaction coefficient required")
    if not all(math.isfinite(v) for v in b + initial):
        raise ValueError("finite vectors required")
    for value in (rtol, atol, step_tol):
        if not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative tolerances required")
    if rtol == atol == 0:
        raise ValueError("at least one residual tolerance must be positive")
    if not math.isfinite(sigma) or not 0 < sigma < 0.5:
        raise ValueError("sigma must lie strictly between zero and one half")
    for count in (maxiter, max_backtracks):
        if type(count) is not int or count < 0:
            raise ValueError("nonnegative integer iteration limits required")
    u = initial.copy()
    bnorm, threshold = norm(b), max(atol, rtol * norm(b))
    history, snapshots, steps, trials = [], [], [], []
    accepted_lambda, step_norm, lin_norm, backtracks = "", "", "", ""
    evaluations, small_step = 1, False
    try:
        f = residual(u, b, coefficient)
    except OverflowError:
        f = [math.inf] * len(u)
    for k in range(maxiter + 1):
        rnorm = norm(f)
        history.append({"k": k, "residual_norm": rnorm,
                        "relative_residual": rnorm / bnorm if bnorm else "",
                        "lambda": accepted_lambda, "step_norm": step_norm,
                        "linear_residual": lin_norm, "backtracks": backtracks,
                        "evaluations": evaluations})
        snapshots.append(u.copy())
        if not math.isfinite(rnorm) or not math.isfinite(rnorm * rnorm):
            status = "nonfinite"
            break
        if rnorm <= threshold:
            status = "converged"
            break
        if small_step:
            status = "stagnation"
            break
        if k == maxiter:
            status = "maxiter"
            break
        try:
            diagonal = jacobian_diagonal(u, coefficient)
            delta = tridiagonal_solve(diagonal, [-v for v in f])
            if not all(math.isfinite(v) for v in delta):
                raise ArithmeticError("nonfinite Newton direction")
            defect = linear_defect(diagonal, delta, f)
        except (ArithmeticError, OverflowError):
            status = "linear_solve_failed"
            break
        lin_norm = norm(defect)
        steps.append({"k": k, "f": f.copy(), "diagonal": diagonal,
                      "delta": delta, "defect": defect})
        phi = 0.5 * rnorm * rnorm
        accepted = False
        for trial in range(1 if method == "full" else max_backtracks + 1):
            lam = 2.0 ** (-trial)
            candidate = [value + lam * change for value, change in zip(u, delta)]
            evaluations += 1
            try:
                candidate_f = residual(candidate, b, coefficient)
                candidate_norm = norm(candidate_f)
            except OverflowError:
                candidate_f, candidate_norm = [math.inf] * len(u), math.inf
            candidate_phi = 0.5 * candidate_norm * candidate_norm
            bound = (1.0 - 2.0 * sigma * lam) * phi
            accepted = math.isfinite(candidate_phi) and (
                method == "full" or candidate_phi <= bound)
            trials.append({"k": k, "trial": trial, "lambda": lam,
                           "residual_norm": candidate_norm, "phi": candidate_phi,
                           "armijo_bound": bound, "accepted": int(accepted)})
            if accepted:
                break
        if not accepted:
            status = "nonfinite" if method == "full" else "line_search_failed"
            break
        step_norm = norm([lam * value for value in delta])
        small_step = step_norm <= step_tol * (1.0 + norm(u)) or candidate == u
        u, f = candidate, candidate_f
        accepted_lambda, backtracks = lam, trial
    return {"u": u, "status": status, "history": history, "snapshots": snapshots,
            "steps": steps, "trials": trials, "threshold": threshold,
            "b_norm": bnorm, "iterations": len(history) - 1,
            "evaluations": evaluations}


def scalar_newton(equation, initial, maxiter, rtol, atol):
    """The scalar counterexamples are full Newton, with residual scale 2."""
    z, history = float(initial), []
    for k in range(maxiter + 1):
        if equation == "sqrt2":
            f, derivative = z * z - 2.0, 2.0 * z
        else:
            f, derivative = z ** 3 - 2.0 * z + 2.0, 3.0 * z * z - 2.0
        history.append({"k": k, "value": z, "residual": f, "derivative": derivative})
        if abs(f) <= max(atol, 2.0 * rtol):
            status = "converged"
            break
        if k == maxiter:
            status = "maxiter"
            break
        if derivative == 0.0:
            status = "zero_derivative"
            break
        z -= f / derivative
    return history, status


def validate(config):
    def number(value, nonnegative=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (not nonnegative or value >= 0))
    for key in ("meshes", "mu_values", "initial_values", "methods"):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError("nonempty unique lists required: " + key)
    if any(type(N) is not int or N < 2 for N in config["meshes"]):
        raise ValueError("meshes must be integers at least two")
    if any(not number(mu, True) for mu in config["mu_values"]):
        raise ValueError("mu values must be finite and nonnegative")
    if any(not number(v) for v in config["initial_values"]):
        raise ValueError("initial values must be finite")
    if any(v not in ("full", "damped") for v in config["methods"]):
        raise ValueError("unknown Newton method")
    for key in ("rtol", "atol", "step_tol"):
        if not number(config[key], True):
            raise ValueError("invalid tolerance: " + key)
    if config["rtol"] == config["atol"] == 0:
        raise ValueError("at least one residual tolerance must be positive")
    if not number(config["sigma"]) or not 0 < config["sigma"] < 0.5:
        raise ValueError("sigma must lie strictly between zero and one half")
    for key in ("maxiter", "max_backtracks"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError("invalid iteration count: " + key)
    names = []
    if not isinstance(config["scalar_cases"], list):
        raise ValueError("scalar_cases must be a list")
    for case in config["scalar_cases"]:
        names.append(case["name"])
        if (not isinstance(case["name"], str) or not case["name"]
                or case["equation"] not in ("sqrt2", "cycle")
                or not number(case["initial"])
                or type(case["maxiter"]) is not int or case["maxiter"] < 0):
            raise ValueError("invalid scalar case")
    if len(set(names)) != len(names):
        raise ValueError("duplicate scalar names")


def write_csv(path, fields, data):
    def formatted(value):
        return format(value, ".17g") if isinstance(value, float) else value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: formatted(row.get(key, "")) for key in fields} for row in data)


def run(config, output):
    validate(config)  # Reject invalid input before creating output files.
    output.mkdir(parents=True, exist_ok=True)
    data = {key: [] for key in ("loads", "history", "iterates", "linear_steps",
                               "trials", "nodes", "summary", "scalar_history",
                               "scalar_summary")}
    common = ("mu", "N", "initial", "method")
    for mu in config["mu_values"]:
        for N in config["meshes"]:
            h = 1.0 / N
            exact = [4.0 * (i / N) * (1.0 - i / N) for i in range(1, N)]
            b = [h * h * (8.0 + mu * value ** 3) for value in exact]
            for i, (load, value) in enumerate(zip(b, exact), 1):
                data["loads"].append({"mu": mu, "N": N, "i": i, "x": i / N,
                                      "b": load, "exact": value})
            for initial in config["initial_values"]:
                for method in config["methods"]:
                    key = dict(zip(common, (mu, N, initial, method)))
                    result = newton(b, mu * h * h, [float(initial)] * (N - 1), method,
                                    **{k: config[k] for k in ("rtol", "atol", "step_tol",
                                       "maxiter", "sigma", "max_backtracks")})
                    for record, u in zip(result["history"], result["snapshots"]):
                        errors = [v - t for v, t in zip(u, exact)]
                        data["history"].append({**key, **record, "error_norm": norm(errors),
                                                "error_max": max(map(abs, errors))})
                        for i, value in enumerate(u, 1):
                            data["iterates"].append({**key, "k": record["k"],
                                                     "i": i, "value": value})
                    for record in result["steps"]:
                        for i in range(N - 1):
                            data["linear_steps"].append({**key, "k": record["k"], "i": i + 1,
                                **{name: record[name][i] for name in ("f", "diagonal",
                                                                    "delta", "defect")}})
                    data["trials"].extend({**key, **record} for record in result["trials"])
                    full_u, full_exact = [0.0] + result["u"] + [0.0], [0.0] + exact + [0.0]
                    for i, (value, target) in enumerate(zip(full_u, full_exact)):
                        data["nodes"].append({**key, "i": i, "x": i / N, "value": value,
                                              "exact": target, "error": value - target})
                    last = data["history"][-1]
                    data["summary"].append({**key, "n": N - 1, "status": result["status"],
                        "iterations": result["iterations"], "evaluations": result["evaluations"],
                        "b_norm": result["b_norm"], "threshold": result["threshold"],
                        "residual_norm": last["residual_norm"],
                        "relative_residual": last["relative_residual"],
                        "error_norm": last["error_norm"], "error_max": last["error_max"],
                        "damped_steps": sum(r["lambda"] not in ("", 1.0)
                                            for r in result["history"]),
                        "min_lambda": min((r["lambda"] for r in result["history"][1:]),
                                          default="")})
    for case in config["scalar_cases"]:
        history, status = scalar_newton(case["equation"], case["initial"], case["maxiter"],
                                        config["rtol"], config["atol"])
        data["scalar_history"].extend({"case": case["name"], **r} for r in history)
        data["scalar_summary"].append({**case, "status": status,
                                       "iterations": len(history) - 1,
                                       "value": history[-1]["value"],
                                       "residual": history[-1]["residual"]})
    fields = {
        "loads": ("mu", "N", "i", "x", "b", "exact"),
        "history": common + ("k", "residual_norm", "relative_residual", "lambda",
                    "step_norm", "linear_residual", "backtracks", "evaluations",
                    "error_norm", "error_max"),
        "iterates": common + ("k", "i", "value"),
        "linear_steps": common + ("k", "i", "f", "diagonal", "delta", "defect"),
        "trials": common + ("k", "trial", "lambda", "residual_norm", "phi",
                            "armijo_bound", "accepted"),
        "nodes": common + ("i", "x", "value", "exact", "error"),
        "summary": common + ("n", "status", "iterations", "evaluations", "b_norm",
                    "threshold", "residual_norm", "relative_residual", "error_norm",
                    "error_max", "damped_steps", "min_lambda"),
        "scalar_history": ("case", "k", "value", "residual", "derivative"),
        "scalar_summary": ("name", "equation", "initial", "maxiter", "status",
                           "iterations", "value", "residual")}
    for name in fields:
        write_csv(output / (name + ".csv"), fields[name], data[name])
    lines = ["pde-030 Newton experiment", "Python " + sys.version.split()[0],
             "Core dependencies: Python standard library",
             "Domain (0,1); homogeneous Dirichlet; n=N-1; all equations scaled by h^2",
             "Exact nodes 4*x*(1-x); diagnostic only, not solver input",
             "F=A*U+mu*h^2*U^3-b; J diagonal 2+3*mu*h^2*U^2, off-diagonals -1",
             "Outer threshold max(atol,rtol*norm(b)); scalar residual scale 2",
             "Halving trials include lambda=1 through 2^(-max_backtracks)",
             "Linear solves: tridiagonal forward elimination and back substitution",
             "one_node_newton() = " + format(one_node_newton(), ".17g"),
             "Configuration: " + json.dumps(config, sort_keys=True)]
    lines.extend(name + " rows: " + str(len(data[name])) for name in fields)
    for r in data["summary"]:
        lines.append("mu={mu} N={N} initial={initial} {method}: {status} "
                     "iterations={iterations} evaluations={evaluations}".format(**r))
    for r in data["scalar_summary"]:
        lines.append("{name}: {status} iterations={iterations} value={value:.17g}".format(**r))
    (output / "run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10] + lines[11:20]))


if __name__ == "__main__":
    article_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=article_dir / "config/problems.json")
    parser.add_argument("--output", type=Path, default=article_dir / "results")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
