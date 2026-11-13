"""ET008-02: the eight NEW N=12 Péclet samples; p=5 is read only from ET008-01.

All solver arithmetic comes from the already audited ET008-01 face/Thomas core.
No PDE solves are performed for the three frozen p=5 cases.
"""
import argparse
import csv
import json
import math
import platform
from pathlib import Path

from run_et008_01 import (
    FIELDS, csv_write, exact, face_coeff, sha, solve_tridiagonal,
)

BASELINE_GIT_BLOBS = {
    "summary.csv": "6239ea268affa50aa95eb057fd7a366c963f0e7f",
    "checks.json": "974e5962391eefc32f53a712c760b5ed0e7946db",
}
BASELINE_SOURCE_COMMIT = "7635e320b8fc68d820d49b188446b74febb6a627"
BASELINE_CORE_SHA256 = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
STAGE_FIELDS = (
    *FIELDS, "c_penultimate", "tv_excess", "central_a_minus",
    "central_a_plus", "source_core_sha256",
)


def git_blob_sha(path):
    import hashlib
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def check_frozen_baseline(baseline, config, core_path):
    """Check frozen ET008-01 identities, without importing or invoking its runner."""
    if sha(core_path) != BASELINE_CORE_SHA256:
        raise RuntimeError("frozen ET008-01 shared solver was modified")
    for name, blob_id in BASELINE_GIT_BLOBS.items():
        if git_blob_sha(baseline / name) != blob_id:
            raise RuntimeError(f"frozen ET008-01 {name} blob changed")
    if (baseline / "source_commit_sha.txt").read_text().strip() != BASELINE_SOURCE_COMMIT:
        raise RuntimeError("frozen ET008-01 source commit changed")
    with (baseline / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checks = json.loads((baseline / "checks.json").read_text(encoding="utf-8"))
    if len(rows) != 3 or checks.get("status") != "PASS" or checks.get("unique_cases_checked") != 3:
        raise RuntimeError("ET008-01 baseline did not pass")
    if [r["case_id"] for r in rows] != [f"P0600-N0012-M{x}" for x in range(3)]:
        raise RuntimeError("unexpected ET008-01 case IDs")
    if any(r["source_config_sha256"] != sha(config) or
           r["source_code_sha256"] != BASELINE_CORE_SHA256 or
           r["source_stage"] != "ET008-01" for r in rows):
        raise RuntimeError("ET008-01 configuration or source identity mismatch")
    return rows


def run_new_case(case, config_sha, solver_sha, core_sha, output):
    case_id, method = case["case_id"], case["method"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    if case["source_stage"] != "ET008-02" or N != 12 or L != 1 or v != 1:
        raise ValueError("ET008-02 must only solve its frozen N=12 cases")
    h, P = L / N, v * L / D
    p = v * h / D
    if abs(p - case["p"]) > 1e-12 or abs(P - case["P"]) > 1e-11 or p >= 5:
        raise ValueError(f"invalid/new baseline case: {case_id}")
    alpha, beta = face_coeff(method, v, D, h)
    lo, mid, hi = -alpha / h, (alpha - beta) / h, beta / h
    U = solve_tridiagonal(N - 1, lo, mid, hi, 0.0, 1.0)
    C = [0.0, *U, 1.0]
    flux = [alpha * C[i] + beta * C[i + 1] for i in range(N)]
    truth = [exact(i * h, L, P) for i in range(N + 1)]
    if not all(math.isfinite(x) for x in C + flux + truth):
        raise ArithmeticError(f"nonfinite ET008-02 output: {case_id}")

    algebraic = (max(abs(lo * C[i-1] + mid * C[i] + hi * C[i+1])
                    for i in range(1, N)) / max(1.0, abs(-hi)))
    c_scale = max(1.0, *(abs(x) for x in C))
    flux_balance = (max(abs(flux[i] - flux[i-1]) for i in range(1, N)) /
                    max(1.0, v * c_scale, D / h * c_scale))
    errors = [abs(a-b) for a,b in zip(C, truth)]
    total_var = sum(abs(C[i+1]-C[i]) for i in range(N))
    ac_left = -D / (h*h) - v / (2*h)
    ac_right = -D / (h*h) + v / (2*h)

    case_dir = output / "raw" / case_id
    csv_write(case_dir / "nodes.csv", ("node_index", "x", "concentration", "reference"),
              [(i, i*h, C[i], truth[i]) for i in range(N+1)])
    csv_write(case_dir / "flux.csv", ("face_index", "x_face", "flux_positive_x"),
              [(i, (i+0.5)*h, flux[i]) for i in range(N)])
    meta = {
        "case_id": case_id, "stage": "ET008-02", "method": method,
        "N": N, "L": L, "v": v, "D": D, "P": P, "p": p, "h": h,
        "solver": "ET008-01 Thomas core; ET008-02 stage assembly",
        "python_version": platform.python_version(), "status": "success",
        "config_sha256": config_sha, "source_sha256": solver_sha,
        "core_sha256": core_sha, "nodes_sha256": sha(case_dir / "nodes.csv"),
        "flux_sha256": sha(case_dir / "flux.csv"),
    }
    (case_dir / "run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "case_id": case_id, "source_stage": "ET008-02",
        "P": P, "p": p, "L": L, "v": v, "D": D, "N": N, "h": h,
        "method": method, "n_unknowns": N-1, "solver_status": "success",
        "algebraic_residual_rel": algebraic, "flux_balance_rel": flux_balance,
        "c_min": min(C), "c_max": max(C), "undershoot": max(0.0, -min(C)),
        "overshoot": max(0.0, max(C)-1.0), "total_variation": total_var,
        "error_linf": max(errors), "error_l1_discrete": h*sum(errors[1:-1]),
        "source_config_sha256": config_sha, "source_code_sha256": solver_sha,
        "c_penultimate": C[-2], "tv_excess": total_var-1.0,
        "central_a_minus": ac_left, "central_a_plus": ac_right,
        "source_core_sha256": core_sha,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    design = json.loads(args.config.read_text(encoding="utf-8"))
    core_path = Path(__file__).with_name("run_et008_01.py")
    baseline_rows = check_frozen_baseline(args.baseline, args.config, core_path)
    cases = [c for c in design["unique_cases"] if c["source_stage"] == "ET008-02"]
    cases.sort(key=lambda c: (c["p"], c["method"]))
    planned = sorted(set(design["stages"][1]["p_values"]) - {5})
    if len(cases) != 24 or len(baseline_rows) != 3 or len(planned) != 8:
        raise RuntimeError("frozen design requires precisely 24 new cases and 3 reuse rows")
    if sorted(set(c["p"] for c in cases)) != planned:
        raise RuntimeError("ET008-02 p values have changed")
    if set(c["method"] for c in cases) != {"M0","M1","M2"}:
        raise RuntimeError("missing method")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refuse to overwrite an existing ET008-02 output")
    args.output.mkdir(parents=True, exist_ok=True)
    config_sha, code_sha, core_sha = sha(args.config), sha(__file__), sha(core_path)
    results = [run_new_case(c, config_sha, code_sha, core_sha, args.output) for c in cases]
    if len({r["case_id"] for r in results}) != 24:
        raise RuntimeError("duplicate ET008-02 case ID")
    csv_write(args.output / "summary.csv", STAGE_FIELDS,
              [[r[k] for k in STAGE_FIELDS] for r in results])
    print(f"ET008-02: solved {len(results)} new N=12 cases; frozen p=5 baseline read, not re-solved.")
    print("central method near p=2:")
    for r in results:
        if r["method"] == "M0" and 1.9 - 1e-12 <= r["p"] <= 2.1 + 1e-12:
            print(f"p={r['p']:.3g}: C11={r['c_penultimate']:.12g}, "
                  f"a_right={r['central_a_plus']:.12g}, "
                  f"TV={r['total_variation']:.12g}")


if __name__ == "__main__":
    main()
