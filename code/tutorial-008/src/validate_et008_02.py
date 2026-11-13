"""Independent ET008-02 audit: exact discrete roots, raw face budgets, frozen reuse.

No imports from either production solver. Never solves any p=5 case.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BASELINE_SOURCE_COMMIT = "7635e320b8fc68d820d49b188446b74febb6a627"
BASELINE_CORE_SHA = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
BASELINE_BLOBS = {
    "summary.csv": "6239ea268affa50aa95eb057fd7a366c963f0e7f",
    "checks.json": "974e5962391eefc32f53a712c760b5ed0e7946db",
    "source_commit_sha.txt": "3f32d769fafad71090b8250af8cff7b0e3585158",
}
RAW_BLOBS = {
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
SCAN = [0.5, 1.0, 1.5, 1.9, 2.0, 2.1, 2.5, 3.0, 5.0]
METHODS = ["M0", "M1", "M2"]
COMPARISON_FIELDS = (
    "case_id", "source_stage", "source_summary_git_blob",
    "P", "p", "D", "N", "method", "c_penultimate", "c_min", "c_max",
    "undershoot", "overshoot", "tv_excess", "error_linf",
    "central_a_minus", "central_a_plus",
    "algebraic_residual_rel", "flux_balance_rel",
    "source_config_sha256", "source_code_sha256",
    "nodes_sha256", "flux_sha256",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path):
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def must_close(a, b, what, atol=5e-11):
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > atol:
        raise AssertionError(f"{what}: got {a}, expected {b}")


def continuous(i, N, P):
    return (math.exp(-P * (1 - i / N)) - math.exp(-P)) / (-math.expm1(-P))


def discrete(i, N, p, method):
    if method == "M2":
        return continuous(i, N, N * p)
    if method == "M0" and p == 2.0:
        return float(i == N)
    base = 1 + p if method == "M1" else (1 + p/2) / (1 - p/2)
    return (base**i - 1) / (base**N - 1)


def coefficients(method, v, D, h):
    if method == "M0":
        return -D / h**2 - v/(2*h), 2*D/h**2, -D/h**2 + v/(2*h)
    if method == "M1":
        return -D/h**2 - v/h, 2*D/h**2 + v/h, -D/h**2
    p = v*h/D
    Bminus = p / (-math.expm1(-p))
    Bplus = p / math.expm1(p)
    return -D/h**2*Bminus, D/h**2*(Bminus+Bplus), -D/h**2*Bplus


def face(method, left, right, v, D, h):
    if method == "M0":
        return v/2*(left+right) - D/h*(right-left)
    if method == "M1":
        return v*left - D/h*(right-left)
    p = v*h/D
    return (D/h)*(p/(-math.expm1(-p))*left - p/math.expm1(p)*right)


def check_frozen_et01(baseline, config_hash):
    for name, pinned in BASELINE_BLOBS.items():
        if git_blob_sha(baseline/name) != pinned:
            raise AssertionError(f"frozen ET008-01 {name} was edited")
    if (baseline/"source_commit_sha.txt").read_text().strip() != BASELINE_SOURCE_COMMIT:
        raise AssertionError("wrong ET008-01 source commit")
    checks = json.loads((baseline/"checks.json").read_text(encoding="utf-8"))
    if checks.get("status") != "PASS" or checks.get("unique_cases_checked") != 3:
        raise AssertionError("frozen ET008-01 checker did not pass")
    check_map = {c["case_id"]: c for c in checks["cases"]}
    rows = read_csv(baseline/"summary.csv")
    if len(rows) != 3 or len(check_map) != 3:
        raise AssertionError("wrong frozen ET008-01 cardinality")
    result = {}
    for row in rows:
        case_id, method = row["case_id"], row["method"]
        if case_id != f"P0600-N0012-{method}" or method not in METHODS:
            raise AssertionError("unexpected ET008-01 case")
        if row["source_stage"] != "ET008-01" or row["solver_status"] != "success":
            raise AssertionError("frozen ET008-01 status changed")
        if row["source_config_sha256"] != config_hash or row["source_code_sha256"] != BASELINE_CORE_SHA:
            raise AssertionError("frozen ET008-01 source or config changed")
        if check_map[case_id]["status"] != "PASS":
            raise AssertionError("frozen ET008-01 case failed")
        case_dir = baseline/"raw"/case_id
        paths = [case_dir/name for name in ("nodes.csv", "flux.csv", "run.json")]
        for path, blob in zip(paths, RAW_BLOBS[method]):
            if git_blob_sha(path) != blob:
                raise AssertionError(f"frozen ET008-01 raw file edited: {path}")
        run = json.loads(paths[2].read_text(encoding="utf-8"))
        if run["nodes_sha256"] != sha256(paths[0]) or run["flux_sha256"] != sha256(paths[1]):
            raise AssertionError("frozen raw content digest failed")
        if run["source_sha256"] != BASELINE_CORE_SHA or run["config_sha256"] != config_hash:
            raise AssertionError("frozen ET008-01 run source mismatch")
        nodes = read_csv(paths[0])
        if len(nodes) != 13:
            raise AssertionError("frozen ET008-01 nodes missing")
        c11 = float(nodes[11]["concentration"])
        must_close(c11, float(check_map[case_id]["c_11"]), "frozen C11")
        must_close(c11, discrete(11, 12, 5.0, method), "frozen analytic C11")
        result[case_id] = (row, run, c11)
    if len(result) != 3:
        raise AssertionError("frozen case IDs not unique")
    return result


def audit_new(case, row, result_dir, config_sha, production_sha, core_sha):
    cid, method = case["case_id"], case["method"]
    N, L, v, D = case["N"], case["L"], case["v"], case["D"]
    h, P, p = L/N, v*L/D, v*(L/N)/D
    if case["source_stage"] != "ET008-02" or N != 12 or (L,v) != (1,1) or p >= 5:
        raise AssertionError("unexpected physical case")
    for field, wanted in {"P":P, "p":p, "D":D, "N":N, "L":L, "v":v, "h":h, "n_unknowns":N-1}.items():
        must_close(float(row[field]), wanted, f"{cid}/{field}")
    if row["case_id"] != cid or row["source_stage"] != "ET008-02" or row["method"] != method:
        raise AssertionError("summary identity mismatch")
    if row["solver_status"] != "success" or row["source_config_sha256"] != config_sha:
        raise AssertionError("solver identity/status mismatch")
    if row["source_code_sha256"] != production_sha or row["source_core_sha256"] != core_sha:
        raise AssertionError("incorrect source SHA in summary")
    folder = result_dir/"raw"/cid
    raw_nodes = read_csv(folder/"nodes.csv")
    raw_faces = read_csv(folder/"flux.csv")
    info = json.loads((folder/"run.json").read_text(encoding="utf-8"))
    if len(raw_nodes) != 13 or len(raw_faces) != 12:
        raise AssertionError("wrong number of nodes or shared faces")
    if any(info[k] != value for k,value in {
        "case_id":cid, "stage":"ET008-02", "method":method,
        "source_sha256":production_sha,"core_sha256":core_sha,
        "config_sha256":config_sha, "status":"success"}.items()):
        raise AssertionError(f"{cid} run.json identity changed")
    if info["nodes_sha256"] != sha256(folder/"nodes.csv") or info["flux_sha256"] != sha256(folder/"flux.csv"):
        raise AssertionError(f"{cid} raw file integrity mismatch")
    C=[]
    for i,node in enumerate(raw_nodes):
        if int(node["node_index"]) != i:
            raise AssertionError("node order mismatch")
        must_close(float(node["x"]), i*h, "node coordinate")
        c=float(node["concentration"])
        ref=continuous(i,N,P)
        must_close(float(node["reference"]),ref,"continuous reference")
        must_close(c,discrete(i,N,p,method),"independent discrete root",atol=1e-10)
        C.append(c)
    if C[0] != 0 or C[-1] != 1:
        raise AssertionError("Dirichlet condition violated")
    F=[]
    for i,face_row in enumerate(raw_faces):
        if int(face_row["face_index"]) != i:
            raise AssertionError("face order mismatch")
        must_close(float(face_row["x_face"]),(i+0.5)*h,"face coordinate")
        fi=float(face_row["flux_positive_x"])
        must_close(fi,face(method,C[i],C[i+1],v,D,h),"independent face",atol=2e-12)
        F.append(fi)
    alo,amid,ahi=coefficients(method,v,D,h)
    RA=(max(abs(alo*C[i-1]+amid*C[i]+ahi*C[i+1]) for i in range(1,N)) /
        max(1.0,abs(ahi)))
    cs=max(1.0,*(abs(c) for c in C))
    RF=max(abs(F[i]-F[i-1]) for i in range(1,N))/max(1.0,v*cs,D/h*cs)
    errs=[abs(C[i]-continuous(i,N,P)) for i in range(N+1)]
    total_var=sum(abs(C[i+1]-C[i]) for i in range(N))
    central_left=-D/h**2-v/(2*h)
    central_right=-D/h**2+v/(2*h)
    metrics={
        "algebraic_residual_rel":RA,"flux_balance_rel":RF,
        "c_min":min(C),"c_max":max(C),"undershoot":max(0.0,-min(C)),
        "overshoot":max(0.0,max(C)-1),"total_variation":total_var,
        "error_linf":max(errs),"error_l1_discrete":h*sum(errs[1:-1]),
        "c_penultimate":C[-2],"tv_excess":total_var-1,
        "central_a_minus":central_left,"central_a_plus":central_right,
    }
    for key,wanted in metrics.items():
        must_close(float(row[key]),wanted,f"{cid}/{key}",atol=1e-10)
    if RA > 1e-10 or RF > 1e-10:
        raise AssertionError("linear algebra or shared-face balance failed")
    if method in {"M1","M2"}:
        if min(C)<-1e-11 or max(C)>1+1e-11 or any(C[i]>C[i+1]+1e-11 for i in range(N)):
            raise AssertionError("upwind/fitted method lost maximum principle")
    if method=="M2" and max(errs)>1e-9:
        raise AssertionError("constant-coefficient exponential fit is inaccurate")
    if method=="M0":
        if p<2:
            if central_right>=0 or min(C)<-1e-11 or any(C[i]>C[i+1]+1e-11 for i in range(N)):
                raise AssertionError("pre-threshold central monotonicity failed")
        elif p==2:
            if abs(central_right)>1e-11 or any(abs(x)>1e-10 for x in C[1:-1]):
                raise AssertionError("p=2 lower-triangular central degeneration failed")
        else:
            if central_right<=0 or C[11]>=-1e-10 or total_var<=1+1e-10:
                raise AssertionError("post-threshold central sign change missing")
    return {
        "case_id":cid,"p":p,"method":method,"status":"PASS",
        "c11":C[-2],"central_a_plus":central_right,"c_min":min(C),
        "tv_excess":total_var-1,"error_linf":max(errs),
        "R_A":RA,"R_F":RF,
        "nodes_sha256":sha256(folder/"nodes.csv"),
        "flux_sha256":sha256(folder/"flux.csv"),
    },info


def fmt(v):
    return format(v,".17g") if isinstance(v,float) else str(v)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--solver-source",type=Path,required=True)
    parser.add_argument("--core-source",type=Path,required=True)
    args=parser.parse_args()
    design=json.loads(args.config.read_text(encoding="utf-8"))
    if design["stages"][1]["p_values"] != SCAN:
        raise AssertionError("Peclet scan changed from reviewed design")
    config_sha,source_sha,core_sha=sha256(args.config),sha256(args.solver_source),sha256(args.core_source)
    if core_sha!=BASELINE_CORE_SHA:
        raise AssertionError("immutable ET008-01 core source changed")
    frozen=check_frozen_et01(args.baseline,config_sha)
    new_cases=sorted((c for c in design["unique_cases"] if c["source_stage"]=="ET008-02"),
                     key=lambda c:(c["p"],c["method"]))
    summaries=read_csv(args.output/"summary.csv")
    if len(new_cases)!=24 or len(summaries)!=24:
        raise AssertionError("must have exactly 24 NEW ET008-02 cases")
    if {c["p"] for c in new_cases}!=set(SCAN)-{5}:
        raise AssertionError("wrong ET008-02 Péclet numbers")
    if {c["case_id"] for c in new_cases} != {s["case_id"] for s in summaries}:
        raise AssertionError("summary has missing or duplicate case IDs")
    if len(list((args.output/"raw").iterdir())) != 24:
        raise AssertionError("ET008-02 raw must have only 24 new case directories")
    summary_git=git_blob_sha(args.output/"summary.csv")
    new_rows={}
    audit=[]
    comparison=[]
    for case,row in zip(new_cases,summaries):
        if row["case_id"]!=case["case_id"]:
            raise AssertionError("summary ordering differs from design")
        checked,info=audit_new(case,row,args.output,config_sha,source_sha,core_sha)
        audit.append(checked)
        new_rows[case["case_id"]]=(row,info)
    for case in sorted((c for c in design["unique_cases"] if "ET008-02" in c["stage_membership"]),
                       key=lambda c:(c["p"],c["method"])):
        cid,method=case["case_id"],case["method"]
        if case["source_stage"]=="ET008-01":
            row,info,c11=frozen[cid]
            src_blob=BASELINE_BLOBS["summary.csv"]
            tv=float(row["total_variation"])
            a_minus=-case["D"]/case["h"]**2-case["v"]/(2*case["h"])
            a_plus=-case["D"]/case["h"]**2+case["v"]/(2*case["h"])
        else:
            row,info=new_rows[cid]
            c11=float(row["c_penultimate"])
            tv=float(row["total_variation"])
            a_minus=float(row["central_a_minus"])
            a_plus=float(row["central_a_plus"])
            src_blob=summary_git
        comparison.append({
            "case_id":cid,"source_stage":case["source_stage"],"source_summary_git_blob":src_blob,
            "P":row["P"],"p":row["p"],"D":row["D"],"N":row["N"],
            "method":method,"c_penultimate":c11,
            "c_min":row["c_min"],"c_max":row["c_max"],
            "undershoot":row["undershoot"],"overshoot":row["overshoot"],
            "tv_excess":tv-1,"error_linf":row["error_linf"],
            "central_a_minus":a_minus,"central_a_plus":a_plus,
            "algebraic_residual_rel":row["algebraic_residual_rel"],
            "flux_balance_rel":row["flux_balance_rel"],
            "source_config_sha256":row["source_config_sha256"],
            "source_code_sha256":row["source_code_sha256"],
            "nodes_sha256":info["nodes_sha256"],
            "flux_sha256":info["flux_sha256"],
        })
    if len(comparison)!=27 or len({r["case_id"] for r in comparison})!=27:
        raise AssertionError("comparison must have exactly 27 unique rows")
    if {r["p"] for r in comparison if r["method"]=="M0"}!={fmt(p) for p in SCAN}:
        # Floats from CSV are formatted with .17g and may display 1.8999999999999999.
        if not all(any(abs(float(r["p"])-p)<1e-12 for r in comparison if r["method"]=="M0") for p in SCAN):
            raise AssertionError("comparison Péclet scan incomplete")
    comparison_path=args.output/"comparison.csv"
    with comparison_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(COMPARISON_FIELDS)
        for row in comparison:
            w.writerow([fmt(row[k]) for k in COMPARISON_FIELDS])
    reused={
        "status":"PASS", "source_stage":"ET008-01",
        "source_commit_sha":BASELINE_SOURCE_COMMIT,
        "source_summary_git_blob":BASELINE_BLOBS["summary.csv"],
        "source_checks_git_blob":BASELINE_BLOBS["checks.json"],
        "reused_case_ids":sorted(frozen),
        "baseline_raw_unchanged":True,
        "new_pde_solves_at_p5":0,
    }
    (args.output/"reused_et008_01.json").write_text(
        json.dumps(reused,indent=2)+"\n",encoding="utf-8")
    summary={
        "stage":"ET008-02","status":"PASS",
        "new_unique_solves_checked":24,"reused_et008_01_cases":3,
        "scan_comparison_count":27,"new_pde_solves_at_p5":0,
        "independent_checker_sha256":sha256(Path(__file__)),
        "new_summary_git_blob":summary_git,
        "case_checks":audit,
    }
    (args.output/"checks.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("ET008-02 independent check PASS: 24 new, 3 ET008-01 reused, 27 comparison points")
    for r in comparison:
        if r["method"]=="M0":
            print(f"p={float(r['p']):.3g} C11={float(r['c_penultimate']):.12g} "
                  f"a_right={float(r['central_a_plus']):.9g} TV-1={float(r['tv_excess']):.9g}")


if __name__=="__main__":
    main()
