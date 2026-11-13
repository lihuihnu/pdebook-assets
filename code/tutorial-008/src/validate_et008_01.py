"""Independent ET008-01 analytic, row and flux audit. Never imports production solver."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(actual, expected, *, atol=5e-11):
    if not math.isfinite(actual) or abs(actual - expected) > atol:
        raise AssertionError(f"numeric mismatch: actual={actual!r}, expected={expected!r}")


def exact_continuous(i, N, P):
    return (math.exp(-P * (1 - i / N)) - math.exp(-P)) / (-math.expm1(-P))


def exact_discrete(i, N, p, method):
    if method == "M2":
        return exact_continuous(i, N, N * p)
    if method == "M1":
        base = 1 + p
    elif method == "M0":
        if p == 2:
            return float(i == N)
        base = (1 + p / 2) / (1 - p / 2)
    else:
        raise AssertionError("unexpected method")
    return (base ** i - 1) / (base ** N - 1)


def independent_coefficients(method, v, D, h):
    if method == "M0":
        return -D / h ** 2 - v / (2 * h), 2 * D / h ** 2, -D / h ** 2 + v / (2 * h)
    if method == "M1":
        return -D / h ** 2 - v / h, 2 * D / h ** 2 + v / h, -D / h ** 2
    p = v * h / D
    bm = -p / math.expm1(-p)
    bp = p / math.expm1(p)
    return -D / h ** 2 * bm, D / h ** 2 * (bm + bp), -D / h ** 2 * bp


def independent_flux(method, left, right, v, D, h):
    if method == "M0":
        return v * (left + right) / 2 - D * (right - left) / h
    if method == "M1":
        return v * left - D * (right - left) / h
    p = v * h / D
    return (D / h) * ((-p / math.expm1(-p)) * left - (p / math.expm1(p)) * right)


def audit_case(case, summary, config_path, solver_path, output):
    case_id = case["case_id"]
    method = case["method"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    h = L / N
    P = v * L / D
    p = v * h / D
    if (N, L, v, P, p) != (12, 1, 1, 60, 5):
        raise AssertionError("unapproved experiment case")
    case_dir = output / "raw" / case_id
    nodes_data = read_csv(case_dir / "nodes.csv")
    flux_data = read_csv(case_dir / "flux.csv")
    run = json.loads((case_dir / "run.json").read_text(encoding="utf-8"))
    if len(nodes_data) != 13 or len(flux_data) != 12:
        raise AssertionError("node and face counts must be 13 and 12")
    if (run["case_id"], run["method"], run["N"], run["status"]) != (case_id, method, N, "success"):
        raise AssertionError("run identity mismatch")
    if run["nodes_sha256"] != digest(case_dir / "nodes.csv") or run["flux_sha256"] != digest(case_dir / "flux.csv"):
        raise AssertionError("raw file hash mismatch")
    if run["config_sha256"] != digest(config_path) or run["source_sha256"] != digest(solver_path):
        raise AssertionError("runner source or config hash mismatch")
    if summary["source_config_sha256"] != run["config_sha256"] or summary["source_code_sha256"] != run["source_sha256"]:
        raise AssertionError("summary identity mismatch")
    if summary["source_stage"] != "ET008-01" or summary["method"] != method or summary["solver_status"] != "success":
        raise AssertionError("summary identity mismatch")
    for key, value in {"N": N, "P": P, "p": p, "L": L, "v": v, "D": D, "h": h, "n_unknowns": N - 1}.items():
        close(float(summary[key]), value)
    C, recorded_flux = [], []
    exact_errors, discrete_errors = [], []
    for i, row in enumerate(nodes_data):
        if int(row["node_index"]) != i:
            raise AssertionError("node order mismatch")
        close(float(row["x"]), i * h)
        val = float(row["concentration"])
        truth = exact_continuous(i, N, P)
        close(float(row["reference"]), truth)
        C.append(val)
        exact_errors.append(abs(val - truth))
        discrete_errors.append(abs(val - exact_discrete(i, N, p, method)))
    if C[0] != 0 or C[-1] != 1:
        raise AssertionError("Dirichlet endpoint changed")
    reconstructed = [independent_flux(method, C[i], C[i + 1], v, D, h) for i in range(N)]
    for i, row in enumerate(flux_data):
        if int(row["face_index"]) != i:
            raise AssertionError("face order mismatch")
        close(float(row["x_face"]), (i + 0.5) * h)
        recorded_flux.append(float(row["flux_positive_x"]))
    flux_diff = max(abs(a - b) for a, b in zip(recorded_flux, reconstructed))
    if flux_diff > 5e-13:
        raise AssertionError(f"independent flux mismatch: {flux_diff}")
    a_left, a_mid, a_right = independent_coefficients(method, v, D, h)
    raw_res = max(abs(a_left * C[i - 1] + a_mid * C[i] + a_right * C[i + 1]) for i in range(1, N))
    algebraic = raw_res / max(1, abs(a_right))
    c_scale = max(1, *(abs(x) for x in C))
    flux_balance = max(abs(recorded_flux[i] - recorded_flux[i - 1]) for i in range(1, N)) / max(1, v * c_scale, D / h * c_scale)
    metrics = {
        "algebraic_residual_rel": algebraic,
        "flux_balance_rel": flux_balance,
        "c_min": min(C), "c_max": max(C),
        "undershoot": max(0.0, -min(C)),
        "overshoot": max(0.0, max(C) - 1),
        "total_variation": sum(abs(C[i + 1] - C[i]) for i in range(N)),
        "error_linf": max(exact_errors),
        "error_l1_discrete": h * sum(exact_errors[1:-1]),
    }
    for key, value in metrics.items():
        close(float(summary[key]), value, atol=1e-10)
    if algebraic > 1e-10 or flux_balance > 1e-10:
        raise AssertionError("algebraic row or shared-face conservation failed")
    max_discrete = max(discrete_errors)
    if max_discrete > 1e-10:
        raise AssertionError("Thomas result differs from independent analytic discrete solution")
    if method == "M0":
        close(C[11], -0.4286262811706461, atol=1e-11)
        if C[11] >= 0 or min(C) >= 0 or metrics["total_variation"] <= 1:
            raise AssertionError("central negative-concentration counterexample missing")
    if method == "M1":
        close(C[11], 0.16666666628383862, atol=1e-11)
    if method == "M2":
        close(C[11], 0.006737946999085467, atol=1e-11)
        if metrics["error_linf"] > 1e-9:
            raise AssertionError("exponential fitting did not reproduce analytic nodes")
    if method in ("M1", "M2"):
        if any(C[i] > C[i + 1] + 1e-12 for i in range(N)) or min(C) < -1e-12 or max(C) > 1 + 1e-12:
            raise AssertionError("expected monotonicity or positivity failed")
    return {"case_id": case_id, "method": method, "max_discrete_node_error": max_discrete,
            "max_independent_flux_difference": flux_diff, "algebraic_residual_rel": algebraic,
            "flux_balance_rel": flux_balance, "c_11": C[11], "c_min": min(C),
            "error_linf": max(exact_errors), "status": "PASS"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--solver-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    design = json.loads(args.config.read_text(encoding="utf-8"))
    cases = sorted((c for c in design["unique_cases"] if c["source_stage"] == "ET008-01"), key=lambda c: c["method"])
    summaries = read_csv(args.output / "summary.csv")
    if len(cases) != 3 or len(summaries) != 3 or [c["method"] for c in cases] != ["M0", "M1", "M2"]:
        raise AssertionError("must audit exactly 3 distinct ET008-01 cases")
    checks = []
    for case, summary in zip(cases, summaries):
        if summary["case_id"] != case["case_id"]:
            raise AssertionError("summary case order mismatch")
        checks.append(audit_case(case, summary, args.config, args.solver_source, args.output))
    result = {"stage": "ET008-01", "status": "PASS", "unique_cases_checked": 3,
              "independent_checker_sha256": digest(Path(__file__)), "cases": checks}
    (args.output / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("ET008-01 independent analytic + flux audit PASS (3/3):")
    for check in checks:
        print(f"{check['method']}: C11={check['c_11']:.12g}, "
              f"R_A={check['algebraic_residual_rel']:.2e}, "
              f"R_F={check['flux_balance_rel']:.2e}")


if __name__ == "__main__":
    main()
