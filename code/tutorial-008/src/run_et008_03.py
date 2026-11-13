"""ET008-03: six NEW N=24/60 fixed-P=60 solves; immutable N=12 reuse.

This production script imports only the frozen ET008-01 face and Thomas core,
never its experiment entrypoint. ET008-01 remains immutable.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

from run_et008_01 import FIELDS, csv_write, exact, face_coeff, sha, solve_tridiagonal

ET01_SOURCE_COMMIT = "7635e320b8fc68d820d49b188446b74febb6a627"
ET01_CORE_SHA = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
ET01_SUMMARY_BLOB = "6239ea268affa50aa95eb057fd7a366c963f0e7f"
ET01_CHECKS_BLOB = "974e5962391eefc32f53a712c760b5ed0e7946db"
ET01_SOURCE_BLOB = "3f32d769fafad71090b8250af8cff7b0e3585158"
ET01_RAW_BLOBS = {
    "M0": ("42867a4af6be9a230b4528dfd83f6d8732f313b3",
           "a7e8e8862ccfdaea391552a0793a67b44674ce23",
           "15700cff8756a25c9e2af498350f3368a2e5f1e6"),
    "M1": ("02a8c12d2b0f28bc756a932bf2c8d0ca7ac62838",
           "812e8e93d2071ec0c85ed04e51c347a448e62e1c",
           "5e5d8c26b993f5363b567e897d0e637062e304bb"),
    "M2": ("5bb889abee1450cd3253fc2301a282e7fa715b98",
           "6a56423ae6d33c1d530ce8cb177635cab552c1d2",
           "68fa44cec3b164f1b182bce81913955de6e04e8d"),
}
METHODS = ("M0", "M1", "M2")
EXTRA_FIELDS = ("c_penultimate", "d_fit", "d_fit_ratio", "d_modified",
                "fit_profile_error_linf", "fit_profile_error_l1_discrete",
                "source_core_sha256")
SUMMARY_FIELDS = FIELDS + EXTRA_FIELDS


def blob_sha(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_et01(baseline, config, core):
    expected = {
        "summary.csv": ET01_SUMMARY_BLOB,
        "checks.json": ET01_CHECKS_BLOB,
        "source_commit_sha.txt": ET01_SOURCE_BLOB,
    }
    if sha(core) != ET01_CORE_SHA:
        raise ValueError("frozen ET008-01 numerical core changed")
    for name, identity in expected.items():
        if blob_sha(baseline / name) != identity:
            raise ValueError(f"frozen ET008-01 identity changed: {name}")
    if (baseline / "source_commit_sha.txt").read_text().strip() != ET01_SOURCE_COMMIT:
        raise ValueError("baseline source commit mismatch")
    checks = json.loads((baseline / "checks.json").read_text())
    if checks.get("status") != "PASS" or checks.get("unique_cases_checked") != 3:
        raise ValueError("frozen baseline checker did not pass")
    rows = read_rows(baseline / "summary.csv")
    if len(rows) != 3 or {r["case_id"] for r in rows} != {
            f"P0600-N0012-{m}" for m in METHODS}:
        raise ValueError("frozen baseline cases differ")
    for row in rows:
        method, case_id = row["method"], row["case_id"]
        if row["source_stage"] != "ET008-01" or row["solver_status"] != "success":
            raise ValueError("old stage or status changed")
        if row["source_code_sha256"] != ET01_CORE_SHA or row["source_config_sha256"] != sha(config):
            raise ValueError("old source/config identity changed")
        directory = baseline / "raw" / case_id
        for name, identity in zip(("nodes.csv", "flux.csv", "run.json"), ET01_RAW_BLOBS[method]):
            if blob_sha(directory / name) != identity:
                raise ValueError(f"old raw data changed: {directory / name}")
    return rows


def solve_new(case, config_sha, source_sha, core_sha, out):
    method, case_id = case["method"], case["case_id"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    h, P, p = L / N, v * L / D, v * (L / N) / D
    if (case["source_stage"] != "ET008-03" or N not in (24, 60) or
            method not in METHODS or (L, v) != (1, 1) or
            abs(P - 60) > 1e-12 or abs(p - case["p"]) > 1e-12):
        raise ValueError(f"not a frozen ET008-03 case: {case_id}")
    alpha, beta = face_coeff(method, v, D, h)
    lower, diag, upper = -alpha / h, (alpha - beta) / h, beta / h
    interior = solve_tridiagonal(N - 1, lower, diag, upper, 0.0, 1.0)
    C = [0.0, *interior, 1.0]
    flux = [alpha * C[i] + beta * C[i + 1] for i in range(N)]
    real = [exact(i * h, L, P) for i in range(N + 1)]

    # Separate physical diffusivity from the fitted/modified diffusivities.
    d_fit = v * h / math.log1p(p)
    d_modified = D + v * h / 2
    P_fit = v * L / d_fit
    fitted = [exact(i * h, L, P_fit) for i in range(N + 1)]
    if not all(math.isfinite(x) for x in C + flux + real + fitted):
        raise ArithmeticError(f"non-finite ET008-03 result: {case_id}")
    err = [abs(x - y) for x, y in zip(C, real)]
    fit_err = [abs(x - y) for x, y in zip(C, fitted)]
    raw_rows = [
        (i, i*h, C[i], real[i], fitted[i]) for i in range(N + 1)
    ]
    folder = out / "raw" / case_id
    csv_write(folder / "nodes.csv",
              ("node_index", "x", "concentration", "reference", "fitted_reference"),
              raw_rows)
    csv_write(folder / "flux.csv",
              ("face_index", "x_face", "flux_positive_x"),
              [(i, (i+.5)*h, flux[i]) for i in range(N)])
    residual = max(abs(lower*C[i-1]+diag*C[i]+upper*C[i+1])
                   for i in range(1, N)) / max(1.0, abs(upper))
    c_scale = max(1.0, *(abs(z) for z in C))
    budget = (max(abs(flux[i]-flux[i-1]) for i in range(1, N)) /
              max(1.0, v*c_scale, (D/h)*c_scale))
    tv = sum(abs(C[i+1]-C[i]) for i in range(N))
    meta = {
        "stage": "ET008-03", "case_id": case_id, "method": method, "status": "success",
        "N": N, "L": L, "v": v, "D": D, "P": P, "p": p, "h": h,
        "solver": "frozen ET008-01 Thomas tridiagonal core",
        "python_version": platform.python_version(),
        "config_sha256": config_sha, "source_sha256": source_sha, "core_sha256": core_sha,
        "nodes_sha256": sha(folder / "nodes.csv"), "flux_sha256": sha(folder / "flux.csv")
    }
    (folder / "run.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    return {
        "case_id": case_id, "source_stage": "ET008-03",
        "P": P, "p": p, "L": L, "v": v, "D": D, "N": N, "h": h, "method": method,
        "n_unknowns": N-1, "solver_status": "success",
        "algebraic_residual_rel": residual, "flux_balance_rel": budget,
        "c_min": min(C), "c_max": max(C),
        "undershoot": max(0., -min(C)), "overshoot": max(0., max(C)-1),
        "total_variation": tv,
        "error_linf": max(err), "error_l1_discrete": h*sum(err[1:-1]),
        "source_config_sha256": config_sha, "source_code_sha256": source_sha,
        "c_penultimate": C[-2], "d_fit": d_fit, "d_fit_ratio": d_fit/D,
        "d_modified": d_modified, "fit_profile_error_linf": max(fit_err),
        "fit_profile_error_l1_discrete": h*sum(fit_err[1:-1]),
        "source_core_sha256": core_sha
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.config.read_text(encoding="utf-8"))
    core_path = Path(__file__).with_name("run_et008_01.py")
    old = verify_et01(args.baseline, args.config, core_path)
    all_cases = [c for c in design["unique_cases"] if "ET008-03" in c["stage_membership"]]
    new_cases = sorted([c for c in all_cases if c["source_stage"] == "ET008-03"],
                       key=lambda c: (c["N"], c["method"]))
    reused = [c for c in all_cases if c["source_stage"] == "ET008-01"]
    expected_new = {f"P0600-N{N:04d}-{m}" for N in (24, 60) for m in METHODS}
    expected_old = {f"P0600-N0012-{m}" for m in METHODS}
    if (len(new_cases) != 6 or {c["case_id"] for c in new_cases} != expected_new or
            len(reused) != 3 or {c["case_id"] for c in reused} != expected_old or
            {c["case_id"] for c in old} != expected_old or len(all_cases) != 9):
        raise ValueError("ET008-03 requires exactly six new and three frozen cases")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refuse to overwrite ET008-03 output")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg_hash, source_hash, core_hash = sha(args.config), sha(__file__), sha(core_path)
    actual = [solve_new(c, cfg_hash, source_hash, core_hash, args.output)
              for c in new_cases]
    csv_write(args.output/"summary.csv", SUMMARY_FIELDS,
              [[r[k] for k in SUMMARY_FIELDS] for r in actual])
    print("ET008-03: solved exactly six NEW N=24/60 cases; N=12 frozen source verified and NOT re-solved")
    for row in actual:
        if row["method"] == "M1":
            print(f"N={row['N']} p={row['p']:.6g} Dfit/D={row['d_fit_ratio']:.12g} "
                  f"fit-node-error={row['fit_profile_error_linf']:.2e} "
                  f"true-node-error={row['error_linf']:.6g}")


if __name__ == "__main__":
    main()
