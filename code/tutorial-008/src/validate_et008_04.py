"""ET008-04 independent verification, immutable provenance and true observed orders.

No production solver is imported. Every one of the 27 profiles is checked
against analytic discrete roots and independently reconstructed face fluxes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ET01_SHA = "7635e320b8fc68d820d49b188446b74febb6a627"
ET03_SHA = "e25b13791e62df4357af1aab9eda3ad41ebdb69a"
ET01_CORE = "d64f8bec5f0dbee653b5f6cb64927ab8ec7c74eb8141119728b176006adbf761"
ET01_SUM = "6239ea268affa50aa95eb057fd7a366c963f0e7f"
ET01_CHECK = "974e5962391eefc32f53a712c760b5ed0e7946db"
ET03_SUM = "9645df86cd4d64f4e0b35e298c8f54e0a4d3e181"
ET03_CHECK = "01dc0ba92db8f1fbfc7975bd74559aa0e759b20c"
ET03_COMPARISON = "b66a7ebba327dd7fb23fbcebce899fcf0bdda8fb"
RAW01 = {
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
NEW_NS = (18,30,48,96,192,384)
ALL_NS = (12,18,24,30,48,60,96,192,384)
METHODS = ("M0","M1","M2")
COMPARISON_FIELDS = (
    "case_id","source_stage","source_summary_git_blob",
    "P","p","L","v","D","N","h","method","n_unknowns",
    "solver_status","c_penultimate","c_min","c_max","undershoot",
    "overshoot","total_variation","error_linf","error_l1_discrete",
    "algebraic_residual_rel","flux_balance_rel",
    "source_config_sha256","source_code_sha256",
    "nodes_sha256","flux_sha256"
)
ORDER_FIELDS = (
    "method","N_coarse","N_fine","h_ratio",
    "error_linf_coarse","error_linf_fine","observed_order_linf",
    "error_l1_coarse","error_l1_fine","observed_order_l1"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_num(actual,expected,label,atol=1e-10):
    if not (math.isfinite(actual) and math.isfinite(expected)) or abs(actual-expected)>atol:
        raise AssertionError(f"{label}: actual {actual}, reference {expected}")


def continuous(i,N,P):
    return (math.exp(-P*(1-i/N))-math.exp(-P)) / (-math.expm1(-P))


def exact_discrete(i,N,p,method):
    if method=="M2":
        return continuous(i,N,N*p)
    if method=="M0" and p==2:
        return float(i==N)
    root = (1+p/2)/(1-p/2) if method=="M0" else (1+p)
    return (root**i-1)/(root**N-1)


def coefficient_row(method,v,D,h):
    if method=="M0":
        return -D/h**2-v/(2*h),2*D/h**2,-D/h**2+v/(2*h)
    if method=="M1":
        return -D/h**2-v/h,2*D/h**2+v/h,-D/h**2
    p=v*h/D
    bm=p/(-math.expm1(-p));bp=p/math.expm1(p)
    return -D/h**2*bm,D/h**2*(bm+bp),-D/h**2*bp


def flux(method,a,b,v,D,h):
    if method=="M0":
        return v*(a+b)/2-D*(b-a)/h
    if method=="M1":
        return v*a-D*(b-a)/h
    p=v*h/D
    return (D/h)*(p/(-math.expm1(-p))*a-p/math.expm1(p)*b)


def row_float(row,key):
    try:
        value=float(row[key])
    except (KeyError,TypeError,ValueError) as e:
        raise AssertionError(f"bad metric {key}") from e
    if not math.isfinite(value):
        raise AssertionError(f"nonfinite metric {key}")
    return value


def check_profile(case,row,folder,config_sha,production_sha,old01=False,old03=False):
    N=case["N"];method=case["method"];cid=case["case_id"]
    L,v,D=case["L"],case["v"],case["D"]
    h=L/N;P=v*L/D;p=v*h/D
    if N not in ALL_NS or (L,v)!=(1,1) or abs(D-1/60)>1e-14 or P!=60:
        raise AssertionError("physical parameters changed")
    if abs(p-case["p"])>1e-12:
        raise AssertionError("Peclet design mismatch")
    if row["case_id"]!=cid or row["method"]!=method or row["solver_status"]!="success":
        raise AssertionError("identity/status differs")
    source=ET01_CORE if old01 else (row["source_code_sha256"] if old03 else production_sha)
    if row["source_config_sha256"]!=config_sha or row["source_code_sha256"]!=source:
        raise AssertionError("raw production provenance mismatch")
    if row["source_stage"]!=case["source_stage"]:
        raise AssertionError("provenance stage mismatch")
    for field,expected in (("N",N),("n_unknowns",N-1),("P",P),("p",p),
                           ("L",L),("v",v),("D",D),("h",h)):
        check_num(row_float(row,field),expected,cid+field)
    nodes=read_csv(folder/"nodes.csv")
    faces=read_csv(folder/"flux.csv")
    meta=json.loads((folder/"run.json").read_text(encoding="utf-8"))
    if len(nodes)!=N+1 or len(faces)!=N:
        raise AssertionError("wrong number of nodes/faces")
    if meta.get("case_id")!=cid or meta.get("method")!=method or meta.get("status")!="success":
        raise AssertionError("case metadata mismatch")
    if (meta["source_sha256"]!=source or meta["config_sha256"]!=config_sha or
            meta["nodes_sha256"]!=digest(folder/"nodes.csv") or
            meta["flux_sha256"]!=digest(folder/"flux.csv")):
        raise AssertionError("raw hashes differ")
    if not old01 and meta["core_sha256"]!=ET01_CORE:
        raise AssertionError("shared solver changed")
    if old01:
        actual=tuple(blob(folder/x) for x in ("nodes.csv","flux.csv","run.json"))
        if actual!=RAW01[method]:
            raise AssertionError("ET008-01 Git blob identity changed")
    C=[];err=[];disc=[]
    for i,item in enumerate(nodes):
        if int(item["node_index"])!=i:
            raise AssertionError("wrong node order")
        check_num(row_float(item,"x"),i*h,"node x",atol=1e-12)
        value=row_float(item,"concentration")
        expected=continuous(i,N,P)
        check_num(row_float(item,"reference"),expected,"continuous truth")
        disc_expected=exact_discrete(i,N,p,method)
        if abs(value-disc_expected)>1e-9:
            raise AssertionError(f"{cid} wrong discrete root at node {i}")
        C.append(value);err.append(abs(value-expected));disc.append(abs(value-disc_expected))
    if C[0]!=0 or C[-1]!=1:
        raise AssertionError("Dirichlet condition altered")
    F=[]
    for i,item in enumerate(faces):
        if int(item["face_index"])!=i:
            raise AssertionError("wrong face order")
        check_num(row_float(item,"x_face"),(i+.5)*h,"face coordinate",atol=1e-12)
        val=row_float(item,"flux_positive_x")
        check_num(val,flux(method,C[i],C[i+1],v,D,h),"face flux")
        F.append(val)
    alo,a0,ahi=coefficient_row(method,v,D,h)
    RA=max(abs(alo*C[i-1]+a0*C[i]+ahi*C[i+1]) for i in range(1,N))/max(1.,abs(ahi))
    cs=max(1.,*(abs(c) for c in C))
    RF=max(abs(F[i]-F[i-1]) for i in range(1,N))/max(1.,v*cs,D/h*cs)
    e_linf=max(err); e_l1=h*sum(err[1:-1])
    tv=sum(abs(C[i+1]-C[i]) for i in range(N))
    calc={"c_min":min(C),"c_max":max(C),"undershoot":max(0.,-min(C)),
          "overshoot":max(0.,max(C)-1),"total_variation":tv,
          "error_linf":e_linf,"error_l1_discrete":e_l1,
          "algebraic_residual_rel":RA,"flux_balance_rel":RF}
    for key,expected in calc.items():
        check_num(row_float(row,key),expected,cid+key)
    if RA>1e-10 or RF>1e-10:
        raise AssertionError("algebraic/flux residual above specified threshold")
    if method=="M2" and e_linf>1e-9:
        raise AssertionError("exponential fit not equal to physical truth")
    if method in ("M1","M2"):
        if min(C)<-1e-11 or max(C)>1+1e-11 or any(C[i]>C[i+1]+1e-11 for i in range(N)):
            raise AssertionError("expected monotonicity violated")
    elif p>2+1e-12:
        if C[N-1]>=0 or tv<=1+1e-10:
            raise AssertionError("central high-Pe undershoot not observed")
    elif abs(p-2)<1e-12:
        if abs(ahi)>1e-10 or any(abs(a)>1e-10 for a in C[1:-1]):
            raise AssertionError("central p=2 degeneracy not observed")
    elif min(C)<-1e-11 or any(C[i]>C[i+1]+1e-11 for i in range(N)):
        raise AssertionError("central resolved profile not monotone")
    return calc,meta,max(disc),max(abs(
        F[i]-flux(method,C[i],C[i+1],v,D,h)) for i in range(N))


def fmt(v):
    return format(v,".17g") if isinstance(v,float) else str(v)


def csv_write(path,fields,records):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(fields)
        for row in records:
            writer.writerow([fmt(row[k]) for k in fields])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--baseline-01",type=Path,required=True)
    ap.add_argument("--baseline-03",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--core",type=Path,required=True)
    args=ap.parse_args()
    if digest(args.core)!=ET01_CORE:
        raise AssertionError("immutable core hash changed")
    for path,refs in (
      (args.baseline_01,{"summary.csv":ET01_SUM,"checks.json":ET01_CHECK}),
      (args.baseline_03,{"summary.csv":ET03_SUM,"checks.json":ET03_CHECK,
                         "comparison.csv":ET03_COMPARISON})):
        for name,expected in refs.items():
            if blob(path/name)!=expected:
                raise AssertionError(f"historical Git blob changed: {path/name}")
    for folder,sha in ((args.baseline_01,ET01_SHA),(args.baseline_03,ET03_SHA)):
        if (folder/"source_commit_sha.txt").read_text().strip()!=sha:
            raise AssertionError("historical source commit changed")
    old1check=json.loads((args.baseline_01/"checks.json").read_text())
    old3check=json.loads((args.baseline_03/"checks.json").read_text())
    if (old1check.get("status")!="PASS" or old1check.get("unique_cases_checked")!=3 or
        old3check.get("status")!="PASS" or old3check.get("new_unique_solves_checked")!=6 or
        old3check.get("new_PDE_solves_at_N12")!=0):
        raise AssertionError("historic scientific verification not PASS")
    design=json.loads(args.config.read_text(encoding="utf-8"))
    config_sha,production_sha=digest(args.config),digest(args.source)
    cases=sorted((c for c in design["unique_cases"] if "ET008-04" in c["stage_membership"]),
                 key=lambda x:(x["N"],x["method"]))
    historic01={r["case_id"]:r for r in read_csv(args.baseline_01/"summary.csv")}
    historic03={r["case_id"]:r for r in read_csv(args.baseline_03/"summary.csv")}
    hist03_refs={r["case_id"]:r for r in read_csv(args.baseline_03/"comparison.csv")}
    fresh={r["case_id"]:r for r in read_csv(args.output/"summary.csv")}
    expect_new={f"P0600-N{N:04d}-{m}" for N in NEW_NS for m in METHODS}
    expect_old={f"P0600-N{N:04d}-{m}" for N in (12,24,60) for m in METHODS}
    if (len(cases)!=27 or len(fresh)!=18 or set(fresh)!=expect_new or
        len(historic01)!=3 or len(historic03)!=6 or
        set(historic01)|set(historic03)!=expect_old or
        len(list((args.output/"raw").iterdir()))!=18):
        raise AssertionError("frozen comparison cases incorrect: expected 18 new + 9 old")
    if [c["N"] for c in cases[::3]] != list(ALL_NS):
        raise AssertionError("grid ordering differs from frozen design")
    summary_blob=blob(args.output/"summary.csv")
    summary=[];checks=[];historical=[]
    for case in cases:
        cid=case["case_id"]
        stage=case["source_stage"]
        if stage=="ET008-01":
            row=historic01[cid];folder=args.baseline_01/"raw"/cid;srcblob=ET01_SUM
        elif stage=="ET008-03":
            row=historic03[cid];folder=args.baseline_03/"raw"/cid;srcblob=ET03_SUM
        elif stage=="ET008-04":
            row=fresh[cid];folder=args.output/"raw"/cid;srcblob=summary_blob
        else:
            raise AssertionError("unexpected stage")
        metrics,meta,disc_err,flux_err=check_profile(
            case,row,folder,config_sha,production_sha,
            old01=stage=="ET008-01",old03=stage=="ET008-03"
        )
        if stage=="ET008-03":
            origin=hist03_refs.get(cid)
            if (not origin or origin["source_summary_git_blob"]!=ET03_SUM or
                    origin["nodes_sha256"]!=meta["nodes_sha256"] or
                    origin["flux_sha256"]!=meta["flux_sha256"]):
                raise AssertionError("ET008-03 original comparison and raw SHA do not match")
        if stage!="ET008-04":
            historical.append(cid)
        summary.append({
           "case_id":cid,"source_stage":stage,
           "source_summary_git_blob":srcblob,
           "P":case["P"],"p":case["p"],"L":case["L"],"v":case["v"],
           "D":case["D"],"N":case["N"],"h":case["h"],"method":case["method"],
           "n_unknowns":case["unknowns"],"solver_status":"success",
           "c_penultimate":row["c_penultimate"] if "c_penultimate" in row else
                            read_csv(folder/"nodes.csv")[-2]["concentration"],
           **{key:row[key] for key in (
               "c_min","c_max","undershoot","overshoot","total_variation",
               "error_linf","error_l1_discrete",
               "algebraic_residual_rel","flux_balance_rel")},
           "source_config_sha256":row["source_config_sha256"],
           "source_code_sha256":row["source_code_sha256"],
           "nodes_sha256":meta["nodes_sha256"],
           "flux_sha256":meta["flux_sha256"],
        })
        checks.append({"case_id":cid,"source_stage":stage,"status":"PASS",
                       "N":case["N"],"method":case["method"],
                       "discrete_node_error_linf":disc_err,
                       "independent_face_error_linf":flux_err,
                       "algebraic_residual_rel":metrics["algebraic_residual_rel"],
                       "flux_balance_rel":metrics["flux_balance_rel"]})
    csv_write(args.output/"comparison.csv",COMPARISON_FIELDS,summary)
    mapped={(int(r["N"]),r["method"]):r for r in summary}
    orders=[]
    # Observed order uses each REAL mesh ratio. Never fit M2 at its roundoff floor.
    for method in ("M0","M1"):
        for cN,fN in zip(ALL_NS[:-1],ALL_NS[1:]):
            coarse,fine=mapped[(cN,method)],mapped[(fN,method)]
            e0,e1=float(coarse["error_linf"]),float(fine["error_linf"])
            l0,l1=float(coarse["error_l1_discrete"]),float(fine["error_l1_discrete"])
            ratio=fN/cN
            if min(e0,e1,l0,l1)<=1e-12:
                raise AssertionError("M0/M1 convergence errors reached unexpected roundoff floor")
            orders.append({
              "method":method,"N_coarse":cN,"N_fine":fN,"h_ratio":ratio,
              "error_linf_coarse":e0,"error_linf_fine":e1,
              "observed_order_linf":math.log(e0/e1)/math.log(ratio),
              "error_l1_coarse":l0,"error_l1_fine":l1,
              "observed_order_l1":math.log(l0/l1)/math.log(ratio)
            })
    csv_write(args.output/"orders.csv",ORDER_FIELDS,orders)
    reuse={
        "status":"PASS","old_PDE_solves_repeated":0,
        "reused_case_count":9,"reused_ids":historical,
        "source_summary_git_blobs":{"ET008-01":ET01_SUM,"ET008-03":ET03_SUM},
        "source_checks_git_blobs":{"ET008-01":ET01_CHECK,"ET008-03":ET03_CHECK},
        "source_commits":{"ET008-01":ET01_SHA,"ET008-03":ET03_SHA},
        "raw_hashes_verified":True,
    }
    (args.output/"reused_histories.json").write_text(
        json.dumps(reuse,indent=2)+"\n",encoding="utf-8")
    report={
      "stage":"ET008-04","status":"PASS",
      "new_unique_solves_checked":18,"historical_reuse_checked":9,
      "comparison_case_count":27,"observed_order_pair_count":16,
      "old_PDE_solves_repeated":0,
      "independent_checker_sha256":digest(Path(__file__)),
      "new_summary_git_blob":summary_blob,
      "cases":checks
    }
    (args.output/"checks.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("ET008-04 independent analytic/flux audit PASS: 18 new, 9 reused, 27 profiles")
    for N in ALL_NS:
        a=mapped[(N,"M0")];b=mapped[(N,"M1")]
        print(f"N={N} p={60/N:.7g} M0_min={float(a['c_min']):.7g} "
              f"M0_error={float(a['error_linf']):.7g} "
              f"M1_error={float(b['error_linf']):.7g}")


if __name__=="__main__":
    main()
