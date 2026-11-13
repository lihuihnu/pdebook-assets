"""ET008-01: three distinct N=12 conservative discretizations. Standard library only."""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

FIELDS = (
    "case_id", "source_stage", "P", "p", "L", "v", "D", "N", "h", "method",
    "n_unknowns", "solver_status", "algebraic_residual_rel", "flux_balance_rel",
    "c_min", "c_max", "undershoot", "overshoot", "total_variation",
    "error_linf", "error_l1_discrete", "source_config_sha256", "source_code_sha256",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(x):
    return format(x, ".17g") if isinstance(x, float) else str(x)


def csv_write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([[fmt(v) for v in row] for row in rows])


def face_coeff(method, v, D, h):
    """F_(i+1/2) = alpha*C_i + beta*C_(i+1)."""
    if method == "M0":
        return v / 2 + D / h, v / 2 - D / h
    if method == "M1":
        return v + D / h, -D / h
    if method == "M2":
        p = v * h / D
        den = -math.expm1(-p)
        q = math.exp(-p)
        return v / den, -v * q / den
    raise ValueError(f"unknown method {method}")


def solve_tridiagonal(n, lower, center, upper, left_value, right_value):
    """Thomas algorithm on the N-1 actual interior unknowns, with boundary elimination."""
    diag = [center] * n
    rhs = [0.0] * n
    rhs[0] -= lower * left_value
    rhs[-1] -= upper * right_value
    sup = [upper] * (n - 1)
    for i in range(1, n):
        multiplier = lower / diag[i - 1]
        diag[i] -= multiplier * sup[i - 1]
        rhs[i] -= multiplier * rhs[i - 1]
    if not all(math.isfinite(a) and abs(a) > 1e-13 for a in diag):
        raise ArithmeticError("nonfinite or nearly zero Thomas pivot")
    result = [0.0] * n
    result[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        result[i] = (rhs[i] - sup[i] * result[i + 1]) / diag[i]
    return result


def exact(x, L, P):
    return (math.exp(-P * (1 - x / L)) - math.exp(-P)) / (-math.expm1(-P))


def run_case(case, config_hash, code_hash, output):
    case_id, method = case["case_id"], case["method"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    h = L / N
    P = v * L / D
    p = v * h / D
    if N != 12 or P != 60 or p != 5 or (L, v) != (1, 1):
        raise ValueError("ET008-01 permits only the frozen N=12, P=60 baseline")
    alpha, beta = face_coeff(method, v, D, h)
    lo, mid, hi = -alpha / h, (alpha - beta) / h, beta / h
    U = solve_tridiagonal(N - 1, lo, mid, hi, 0.0, 1.0)
    C = [0.0, *U, 1.0]
    flux = [alpha * C[i] + beta * C[i + 1] for i in range(N)]
    if not all(math.isfinite(y) for y in C + flux):
        raise ArithmeticError("nonfinite concentration or flux")
    reference = [exact(i * h, L, P) for i in range(N + 1)]
    error = [abs(a - b) for a, b in zip(C, reference)]
    b_last = -hi * 1.0
    residual = []
    for i in range(1, N):
        lhs = mid * C[i] + lo * C[i - 1] + hi * C[i + 1]
        residual.append(lhs)
    # Algebraic residual computed before boundary elimination; equivalent to A*U-b.
    algebraic = max(abs(y) for y in residual) / max(1.0, abs(b_last))
    scale = max(1.0, v * max(1.0, max(abs(y) for y in C)),
                D / h * max(1.0, max(abs(y) for y in C)))
    balance = max(abs(flux[i] - flux[i - 1]) for i in range(1, N)) / scale
    metrics = {
        "case_id": case_id, "source_stage": "ET008-01", "P": P, "p": p,
        "L": L, "v": v, "D": D, "N": N, "h": h, "method": method,
        "n_unknowns": N - 1, "solver_status": "success",
        "algebraic_residual_rel": algebraic, "flux_balance_rel": balance,
        "c_min": min(C), "c_max": max(C), "undershoot": max(0.0, -min(C)),
        "overshoot": max(0.0, max(C) - 1),
        "total_variation": sum(abs(C[i + 1] - C[i]) for i in range(N)),
        "error_linf": max(error), "error_l1_discrete": h * sum(error[1:-1]),
        "source_config_sha256": config_hash, "source_code_sha256": code_hash,
    }
    case_dir = output / "raw" / case_id
    csv_write(case_dir / "nodes.csv", ("node_index", "x", "concentration", "reference"),
              [(i, i * h, C[i], reference[i]) for i in range(N + 1)])
    csv_write(case_dir / "flux.csv", ("face_index", "x_face", "flux_positive_x"),
              [(i, (i + 0.5) * h, flux[i]) for i in range(N)])
    record = {
        "case_id": case_id, "stage": "ET008-01", "method": method,
        "N": N, "P": P, "p": p, "L": L, "v": v, "D": D, "h": h,
        "python_version": platform.python_version(), "solver": "explicit Thomas tridiagonal",
        "config_sha256": config_hash, "source_sha256": code_hash,
        "nodes_sha256": sha(case_dir / "nodes.csv"),
        "flux_sha256": sha(case_dir / "flux.csv"),
        "status": "success",
    }
    (case_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    design = json.loads(args.config.read_text(encoding="utf-8"))
    cases = [c for c in design["unique_cases"] if c["source_stage"] == "ET008-01"]
    cases.sort(key=lambda c: c["method"])
    if len(cases) != 3 or [c["method"] for c in cases] != ["M0", "M1", "M2"]:
        raise ValueError("ET008-01 must contain exactly three unique frozen cases")
    args.output.mkdir(parents=True, exist_ok=True)
    config_hash = sha(args.config)
    code_hash = sha(__file__)
    rows = [run_case(c, config_hash, code_hash, args.output) for c in cases]
    csv_write(args.output / "summary.csv", FIELDS, [[r[k] for k in FIELDS] for r in rows])
    print("ET008-01 solver produced exactly three N=12 cases:")
    for item in rows:
        print(f"{item['method']}: min={item['c_min']:.10g}, max={item['c_max']:.10g}, "
              f"error_linf={item['error_linf']:.4g}, R_A={item['algebraic_residual_rel']:.2e}, "
              f"R_F={item['flux_balance_rel']:.2e}")


if __name__ == "__main__":
    main()
