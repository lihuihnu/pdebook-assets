"""ET008-04: exactly 18 new fixed-physics N=18/30/48/96/192/384 solves.

The frozen ET008-01 numerical core is reused; N=12/24/60 histories are
read and hash-checked, never passed to the Thomas solver.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

from run_et008_01 import FIELDS, csv_write, exact, face_coeff, sha, solve_tridiagonal

NEW_N = (18, 30, 48, 96, 192, 384)
OLD_N = (12, 24, 60)
METHODS = ("M0", "M1", "M2")
ET01_CORE_SHA = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
ET01_SUMMARY_BLOB = "6239ea268affa50aa95eb057fd7a366c963f0e7f"
ET01_CHECKS_BLOB = "974e5962391eefc32f53a712c760b5ed0e7946db"
ET01_COMMIT = "7635e320b8fc68d820d49b188446b74febb6a627"
ET03_SUMMARY_BLOB = "9645df86cd4d64f4e0b35e298c8f54e0a4d3e181"
ET03_COMPARE_BLOB = "b66a7ebba327dd7fb23fbcebce899fcf0bdda8fb"
ET03_CHECKS_BLOB = "01dc0ba92db8f1fbfc7975bd74559aa0e759b20c"
ET03_COMMIT = "e25b13791e62df4357af1aab9eda3ad41ebdb69a"
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
EXTRA_FIELDS = ("c_penultimate", "source_core_sha256")
SUMMARY_FIELDS = FIELDS + EXTRA_FIELDS


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def inspect_history(folder, stage, summary_blob, checks_blob, source_commit,
                    allowed_n, compare_blob=None):
    if git_blob(folder / "summary.csv") != summary_blob:
        raise ValueError(f"{stage} frozen summary blob mismatch")
    if git_blob(folder / "checks.json") != checks_blob:
        raise ValueError(f"{stage} frozen checks blob mismatch")
    if (folder / "source_commit_sha.txt").read_text().strip() != source_commit:
        raise ValueError(f"{stage} frozen source commit mismatch")
    check = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
    if check.get("status") != "PASS":
        raise ValueError(f"{stage} independent check is not PASS")
    if compare_blob is not None and git_blob(folder / "comparison.csv") != compare_blob:
        raise ValueError(f"{stage} frozen comparison blob mismatch")
    previous = csv_rows(folder / "summary.csv")
    expect = {f"P0600-N{N:04d}-{m}" for N in allowed_n for m in METHODS}
    if {r["case_id"] for r in previous} != expect or len(previous) != len(expect):
        raise ValueError(f"{stage} historical case set changed")
    refs = {}
    if compare_blob is not None:
        refs = {r["case_id"]: r for r in csv_rows(folder / "comparison.csv")}
        if len(refs) != 9 or check.get("new_unique_solves_checked") != 6:
            raise ValueError("ET008-03 independent history incomplete")
    for row in previous:
        case_id = row["case_id"]
        method = row["method"]
        if row["source_stage"] != stage or row["solver_status"] != "success":
            raise ValueError(f"{stage} provenance mismatch")
        raw = folder / "raw" / case_id
        paths = [raw / n for n in ("nodes.csv", "flux.csv", "run.json")]
        meta = json.loads(paths[2].read_text(encoding="utf-8"))
        if (meta["nodes_sha256"] != sha(paths[0]) or
                meta["flux_sha256"] != sha(paths[1]) or
                meta["source_sha256"] != row["source_code_sha256"]):
            raise ValueError(f"{stage} raw SHA mismatch: {case_id}")
        if stage == "ET008-01":
            if tuple(git_blob(p) for p in paths) != ET01_RAW_BLOBS[method]:
                raise ValueError(f"ET008-01 raw Git blob mismatch: {case_id}")
        else:
            ref = refs.get(case_id)
            if (not ref or ref["source_stage"] != stage or
                    ref["source_summary_git_blob"] != summary_blob or
                    ref["nodes_sha256"] != meta["nodes_sha256"] or
                    ref["flux_sha256"] != meta["flux_sha256"]):
                raise ValueError(f"ET008-03 comparison/raw mismatch: {case_id}")
    return previous


def solve_new(case, cfg_sha, code_sha, core_sha, root):
    cid, method = case["case_id"], case["method"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    h = L / N
    P = v * L / D
    p = v * h / D
    if (case["source_stage"] != "ET008-04" or N not in NEW_N or
            method not in METHODS or (L, v) != (1, 1) or
            abs(D - 1/60) > 1e-15 or abs(P - 60) > 1e-12 or
            abs(p - case["p"]) > 1e-12 or cid != f"P0600-N{N:04d}-{method}"):
        raise ValueError(f"ET008-04 disallowed case: {cid}")
    left, right = face_coeff(method, v, D, h)
    a_minus, a_diag, a_plus = -left/h, (left-right)/h, right/h
    U = solve_tridiagonal(N-1, a_minus, a_diag, a_plus, 0., 1.)
    C = [0., *U, 1.]
    fluxes = [left*C[i] + right*C[i+1] for i in range(N)]
    ref = [exact(i*h, L, P) for i in range(N+1)]
    if not all(math.isfinite(y) for y in C+fluxes+ref):
        raise ArithmeticError(f"non-finite numerical output: {cid}")
    err = [abs(x-y) for x,y in zip(C,ref)]
    ra = max(abs(a_minus*C[i-1]+a_diag*C[i]+a_plus*C[i+1])
             for i in range(1,N))/max(1.,abs(a_plus))
    cs=max(1.,*(abs(c) for c in C))
    rf = max(abs(fluxes[i]-fluxes[i-1]) for i in range(1,N)) / max(1.,v*cs,D/h*cs)
    out=root/"raw"/cid
    csv_write(out/"nodes.csv",("node_index","x","concentration","reference"),
              [(i,i*h,C[i],ref[i]) for i in range(N+1)])
    csv_write(out/"flux.csv",("face_index","x_face","flux_positive_x"),
              [(i,(i+.5)*h,fluxes[i]) for i in range(N)])
    meta={
        "stage":"ET008-04","case_id":cid,"method":method,"N":N,
        "L":L,"v":v,"D":D,"P":P,"p":p,"h":h,
        "solver":"ET008-01 frozen Thomas/face core",
        "python_version":platform.python_version(),
        "config_sha256":cfg_sha,"source_sha256":code_sha,"core_sha256":core_sha,
        "nodes_sha256":sha(out/"nodes.csv"),"flux_sha256":sha(out/"flux.csv"),
        "status":"success"
    }
    (out/"run.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    return {
        "case_id":cid,"source_stage":"ET008-04",
        "P":P,"p":p,"L":L,"v":v,"D":D,"N":N,"h":h,"method":method,
        "n_unknowns":N-1,"solver_status":"success",
        "algebraic_residual_rel":ra,"flux_balance_rel":rf,
        "c_min":min(C),"c_max":max(C),
        "undershoot":max(0.,-min(C)),"overshoot":max(0.,max(C)-1),
        "total_variation":sum(abs(C[i+1]-C[i]) for i in range(N)),
        "error_linf":max(err),"error_l1_discrete":h*sum(err[1:-1]),
        "source_config_sha256":cfg_sha,"source_code_sha256":code_sha,
        "c_penultimate":C[-2],"source_core_sha256":core_sha,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--baseline-01",type=Path,required=True)
    parser.add_argument("--baseline-03",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    core=Path(__file__).with_name("run_et008_01.py")
    if sha(core)!=ET01_CORE_SHA:
        raise ValueError("previously validated numerical core changed")
    old1=inspect_history(args.baseline_01,"ET008-01",ET01_SUMMARY_BLOB,
                         ET01_CHECKS_BLOB,ET01_COMMIT,(12,))
    old3=inspect_history(args.baseline_03,"ET008-03",ET03_SUMMARY_BLOB,
                         ET03_CHECKS_BLOB,ET03_COMMIT,(24,60),ET03_COMPARE_BLOB)
    design=json.loads(args.config.read_text(encoding="utf-8"))
    stage=next(s for s in design["stages"] if s["stage_id"]=="ET008-04")
    if stage["N_values"] != [12,18,24,30,48,60,96,192,384] or stage["global_peclet"] != 60:
        raise ValueError("ET008-04 registered experiment design changed")
    members=[c for c in design["unique_cases"] if "ET008-04" in c["stage_membership"]]
    newer=sorted([c for c in members if c["source_stage"]=="ET008-04"],
                 key=lambda x:(x["N"],x["method"]))
    expected_new={f"P0600-N{N:04d}-{m}" for N in NEW_N for m in METHODS}
    expected_old={f"P0600-N{N:04d}-{m}" for N in OLD_N for m in METHODS}
    if (len(members)!=27 or len(newer)!=18 or
            {c["case_id"] for c in newer}!=expected_new or
            {c["case_id"] for c in members if c["source_stage"]!="ET008-04"}!=expected_old or
            len(old1)!=3 or len(old3)!=6):
        raise ValueError("ET008-04 must contain only 18 new + 9 read-only cases")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refuse to overwrite an existing output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    cfg,code=sha(args.config),sha(__file__)
    results=[solve_new(c,cfg,code,ET01_CORE_SHA,args.output) for c in newer]
    csv_write(args.output/"summary.csv", SUMMARY_FIELDS,
              [[r[k] for k in SUMMARY_FIELDS] for r in results])
    print("ET008-04: solved 18 new cases; N=12,24,60 frozen histories read only; zero old reruns")
    for r in results:
        if r["method"]=="M0":
            print(f"N={r['N']} pe={r['p']:.8g} min={r['c_min']:.8g} "
                  f"TV={r['total_variation']:.8g} errorLinf={r['error_linf']:.8g}")


if __name__=="__main__":
    main()
