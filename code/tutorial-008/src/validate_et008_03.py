"""Independent ET008-03 checker. No imports from any production solver.

Derives nodal closed forms, Bernoulli flux, the exact upwind fitted
diffusivity, and the fixed physical solution independently. N=12 is read only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ET01_COMMIT = "7635e320b8fc68d820d49b188446b74febb6a627"
ET01_CORE_SHA = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
ET01_BLOBS = {
    "summary.csv": "6239ea268affa50aa95eb057fd7a366c963f0e7f",
    "checks.json": "974e5962391eefc32f53a712c760b5ed0e7946db",
    "source_commit_sha.txt": "3f32d769fafad71090b8250af8cff7b0e3585158",
}
ET01_RAW = {
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
COMPARISON_FIELDS = (
    "case_id", "source_stage", "source_summary_git_blob", "N", "P", "p",
    "D", "h", "method", "c_penultimate", "c_min", "c_max",
    "undershoot", "overshoot", "total_variation", "error_linf",
    "algebraic_residual_rel", "flux_balance_rel",
    "d_fit", "d_fit_ratio", "d_modified",
    "fit_profile_error_linf", "fit_profile_error_l1_discrete",
    "source_config_sha256", "source_code_sha256",
    "nodes_sha256", "flux_sha256"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as fd:
        return list(csv.DictReader(fd))


def close(value, target, name, tol=1e-10):
    if not math.isfinite(value) or abs(value-target)>tol:
        raise AssertionError(f"{name}: {value!r} != {target!r}")


def exact(i, N, total_pe):
    return (math.exp(-total_pe*(1.0-i/N))-math.exp(-total_pe)) / (-math.expm1(-total_pe))


def discrete(i, N, pe, method):
    if method == "M2":
        return exact(i,N,N*pe)
    if method == "M0":
        if pe == 2.0:
            return float(i == N)
        root=(1+pe/2)/(1-pe/2)
    elif method == "M1":
        root=1+pe
    else:
        raise AssertionError("unknown method")
    return (root**i-1)/(root**N-1)


def dfit(h, v, p):
    return v*h/math.log1p(p)


def stencil(method,v,D,h):
    if method == "M0":
        return -D/h**2-v/(2*h),2*D/h**2,-D/h**2+v/(2*h)
    if method == "M1":
        return -D/h**2-v/h,2*D/h**2+v/h,-D/h**2
    p=v*h/D
    bm=p/(-math.expm1(-p))
    bp=p/math.expm1(p)
    return -D/h**2*bm,D/h**2*(bm+bp),-D/h**2*bp


def face(method,left,right,v,D,h):
    if method == "M0":
        return v*(left+right)/2-D*(right-left)/h
    if method == "M1":
        return v*left-D*(right-left)/h
    p=v*h/D
    return (D/h)*(p/(-math.expm1(-p))*left-p/math.expm1(p)*right)


def validate_nodes(case, directory, summary_row, *, has_fit_column):
    method,N,L,v,D=case["method"],case["N"],case["L"],case["v"],case["D"]
    h=L/N;P=v*L/D;p=v*h/D
    if P!=60 or (L,v)!=(1,1) or N not in (12,24,60):
        raise AssertionError("unexpected physical model")
    for field,expected in (("N",N),("P",P),("p",p),("D",D),("h",h),
                            ("n_unknowns",N-1)):
        if field in summary_row:
            close(float(summary_row[field]),expected,f"{method} N{N}/{field}")
    if summary_row["case_id"] != case["case_id"] or summary_row["method"] != method:
        raise AssertionError("summary/model identity mismatch")
    node_file=directory/"nodes.csv"
    face_file=directory/"flux.csv"
    node_rows=rows(node_file);face_rows=rows(face_file)
    meta=json.loads((directory/"run.json").read_text(encoding="utf-8"))
    if (len(node_rows),len(face_rows))!=(N+1,N):
        raise AssertionError("node/face cardinality mismatch")
    if meta["case_id"]!=case["case_id"] or meta["method"]!=method or meta["status"]!="success":
        raise AssertionError("run metadata mismatch")
    if meta["nodes_sha256"]!=sha(node_file) or meta["flux_sha256"]!=sha(face_file):
        raise AssertionError("raw content failed SHA-256 check")
    fit_D=dfit(h,v,p)
    fit_P=v*L/fit_D
    fit_err=[];real_err=[];discrete_err=[]
    C=[]
    for i,item in enumerate(node_rows):
        if int(item["node_index"])!=i:
            raise AssertionError("unordered nodes")
        close(float(item["x"]),i*h,"node coordinate")
        value=float(item["concentration"])
        truth=exact(i,N,P)
        fit=exact(i,N,fit_P)
        close(float(item["reference"]),truth,"physical reference",tol=1e-12)
        if has_fit_column:
            close(float(item["fitted_reference"]),fit,"fitted reference",tol=1e-12)
        C.append(value)
        real_err.append(abs(value-truth))
        fit_err.append(abs(value-fit))
        discrete_err.append(abs(value-discrete(i,N,p,method)))
    if C[0]!=0 or C[-1]!=1 or not all(math.isfinite(c) for c in C):
        raise AssertionError("Dirichlet or finite-value failure")
    if max(discrete_err)>1e-10:
        raise AssertionError("production nodes fail independent discrete roots")
    face_values=[]
    for i,item in enumerate(face_rows):
        if int(item["face_index"])!=i:
            raise AssertionError("unordered faces")
        close(float(item["x_face"]),(i+.5)*h,"face coordinate")
        value=float(item["flux_positive_x"])
        close(value,face(method,C[i],C[i+1],v,D,h),"independent flux",tol=1e-11)
        face_values.append(value)
    alo,a0,ahi=stencil(method,v,D,h)
    ra=max(abs(alo*C[i-1]+a0*C[i]+ahi*C[i+1]) for i in range(1,N))/max(1.,abs(ahi))
    cs=max(1.,*(abs(c) for c in C))
    rf=max(abs(face_values[i]-face_values[i-1]) for i in range(1,N))/max(1.,v*cs,D/h*cs)
    total_var=sum(abs(C[i+1]-C[i]) for i in range(N))
    computed={
        "c_min":min(C),"c_max":max(C),"undershoot":max(0.,-min(C)),
        "overshoot":max(0.,max(C)-1.),"total_variation":total_var,
        "error_linf":max(real_err),
        "error_l1_discrete":h*sum(real_err[1:-1]),
        "algebraic_residual_rel":ra,"flux_balance_rel":rf,
        "c_penultimate":C[-2],
        "d_fit":fit_D, "d_fit_ratio":fit_D/D,
        "d_modified":D+v*h/2,
        "fit_profile_error_linf":max(fit_err),
        "fit_profile_error_l1_discrete":h*sum(fit_err[1:-1])
    }
    for key,expected in computed.items():
        if key in summary_row:
            close(float(summary_row[key]),expected,f"{case['case_id']}/{key}",tol=1e-10)
    if ra>1e-10 or rf>1e-10:
        raise AssertionError("discrete algebraic or face-budget mismatch")
    if not (fit_D>D and fit_D/D>1):
        raise AssertionError("fitted diffusion should exceed physical diffusion")
    if method=="M1":
        if max(fit_err)>1e-10:
            raise AssertionError("upwind profile disagrees with fitted continuum solution")
        if any(C[i]>C[i+1]+1e-11 for i in range(N)) or min(C)<-1e-11:
            raise AssertionError("upwind loses nonnegativity or monotonicity")
    elif method=="M2":
        if max(real_err)>1e-9:
            raise AssertionError("exponential fit does not recover physical analytic nodes")
    elif method=="M0" and N==24 and C[N-1]>=0:
        raise AssertionError("underresolved central-difference negative node missing")
    elif method=="M0" and N==60 and min(C)<-1e-11:
        raise AssertionError("resolved central-difference negative node unexpected")
    return computed,meta,max(discrete_err),max(abs(
        face_values[i]-face(method,C[i],C[i+1],v,D,h)) for i in range(N))


def fmt(v):
    return format(v,".17g") if isinstance(v,float) else str(v)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--core",type=Path,required=True)
    args=p.parse_args()
    design=json.loads(args.config.read_text(encoding="utf-8"))
    if sha(args.core)!=ET01_CORE_SHA:
        raise AssertionError("baseline numerical core modified")
    for name,want in ET01_BLOBS.items():
        if blob(args.baseline/name)!=want:
            raise AssertionError("ET008-01 source file no longer matches recorded Git blob")
    if (args.baseline/"source_commit_sha.txt").read_text().strip()!=ET01_COMMIT:
        raise AssertionError("ET008-01 provenance changed")
    baseline_checks=json.loads((args.baseline/"checks.json").read_text(encoding="utf-8"))
    if baseline_checks["status"]!="PASS" or baseline_checks["unique_cases_checked"]!=3:
        raise AssertionError("ET008-01 independent check was not successful")
    baseline={r["case_id"]:r for r in rows(args.baseline/"summary.csv")}
    new={r["case_id"]:r for r in rows(args.output/"summary.csv")}
    cases=sorted([c for c in design["unique_cases"] if "ET008-03" in c["stage_membership"]],
                 key=lambda c:(c["N"],c["method"]))
    new_ids={f"P0600-N{N:04d}-{m}" for N in (24,60) for m in METHODS}
    old_ids={f"P0600-N0012-{m}" for m in METHODS}
    if (len(cases)!=9 or len(new)!=6 or set(new)!=new_ids or
        len(baseline)!=3 or set(baseline)!=old_ids or
        len(list((args.output/"raw").iterdir()))!=6):
        raise AssertionError("ET008-03 case count must be 6 new + 3 reused")
    if [c["case_id"] for c in cases] != [f"P0600-N{N:04d}-{m}"
            for N in (12,24,60) for m in METHODS]:
        raise AssertionError("wrong case order or membership")
    config_sha,src_sha=sha(args.config),sha(args.source)
    checks=[]; comparison=[]; reuse_ids=[]
    summary_new_blob=blob(args.output/"summary.csv")
    for case in cases:
        cid=case["case_id"]
        is_old=case["source_stage"]=="ET008-01"
        if case["source_stage"]!=("ET008-01" if case["N"]==12 else "ET008-03"):
            raise AssertionError("unexpected stage/source association")
        row=(baseline if is_old else new)[cid]
        folder=(args.baseline if is_old else args.output)/"raw"/cid
        if row["source_config_sha256"]!=config_sha or (
                row["source_code_sha256"] != (ET01_CORE_SHA if is_old else src_sha)):
            raise AssertionError("config/source identity changed")
        meta=json.loads((folder/"run.json").read_text(encoding="utf-8"))
        if (meta["config_sha256"]!=config_sha or
                meta["source_sha256"]!=(ET01_CORE_SHA if is_old else src_sha) or
                (not is_old and meta["core_sha256"]!=ET01_CORE_SHA)):
            raise AssertionError("raw production provenance mismatch")
        if is_old:
            reuse_ids.append(cid)
            for name, expected in zip(("nodes.csv","flux.csv","run.json"),ET01_RAW[case["method"]]):
                if blob(folder/name)!=expected:
                    raise AssertionError(f"frozen ET008-01 raw blob changed: {cid}/{name}")
        else:
            if row["source_stage"]!="ET008-03" or row["source_core_sha256"]!=ET01_CORE_SHA:
                raise AssertionError("new case stage/core digest mismatch")
        measured,_,discrete_error,flux_error=validate_nodes(
            case,folder,row,has_fit_column=not is_old)
        record={
            "case_id":cid,"source_stage":case["source_stage"],
            "source_summary_git_blob":ET01_BLOBS["summary.csv"] if is_old else summary_new_blob,
            "N":case["N"],"P":case["P"],"p":case["p"],"D":case["D"],"h":case["h"],
            "method":case["method"],
            **{k:measured[k] for k in (
              "c_penultimate","c_min","c_max","undershoot","overshoot",
              "total_variation","error_linf","algebraic_residual_rel",
              "flux_balance_rel","d_fit","d_fit_ratio","d_modified",
              "fit_profile_error_linf","fit_profile_error_l1_discrete")},
            "source_config_sha256":row["source_config_sha256"],
            "source_code_sha256":row["source_code_sha256"],
            "nodes_sha256":meta["nodes_sha256"],
            "flux_sha256":meta["flux_sha256"]
        }
        comparison.append(record)
        checks.append({
          "case_id":cid, "status":"PASS",
          "source_stage":case["source_stage"], "N":case["N"],"method":case["method"],
          "max_independent_discrete_node_error":discrete_error,
          "max_independent_face_flux_error":flux_error,
          "algebraic_residual_rel":measured["algebraic_residual_rel"],
          "flux_balance_rel":measured["flux_balance_rel"],
          "fit_profile_error_linf":measured["fit_profile_error_linf"],
          "error_linf":measured["error_linf"]
        })
    with (args.output/"comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream,lineterminator="\n")
        writer.writerow(COMPARISON_FIELDS)
        for record in comparison:
            writer.writerow([fmt(record[key]) for key in COMPARISON_FIELDS])
    reuse={
      "status":"PASS", "source_stage":"ET008-01",
      "source_commit_sha":ET01_COMMIT,
      "source_summary_git_blob":ET01_BLOBS["summary.csv"],
      "source_checks_git_blob":ET01_BLOBS["checks.json"],
      "reused_case_ids":reuse_ids,
      "new_PDE_solves_at_N12":0,
      "baseline_raw_unchanged":True
    }
    (args.output/"reused_et008_01.json").write_text(
        json.dumps(reuse,indent=2)+"\n",encoding="utf-8")
    result={
      "stage":"ET008-03","status":"PASS",
      "new_unique_solves_checked":6,"reused_et008_01_cases":3,
      "comparison_count":9,"new_PDE_solves_at_N12":0,
      "independent_checker_sha256":sha(Path(__file__)),
      "new_summary_git_blob":summary_new_blob,
      "cases":checks
    }
    (args.output/"checks.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("ET008-03 independent audit PASS: six new N24/N60 solves + three immutable N12 reuse")
    for r in comparison:
        if r["method"]=="M1":
            print(f"N={r['N']} p={r['p']:.6g} Dfit={r['d_fit']:.12g} "
                  f"Dfit/D={r['d_fit_ratio']:.12g} "
                  f"fitError={r['fit_profile_error_linf']:.3e} "
                  f"trueError={r['error_linf']:.9g}")


if __name__ == "__main__":
    main()
