"""Independent fail-closed ET005-01 audit; never imports the tested solver."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

TIMES = (0.0, 0.3, 0.6, 0.7, 0.8)
REF_TIMES = TIMES[:3]
N = 200
DX = 1.0 / 200.0
CFL = 0.8
T_BREAK = 2.0 / math.pi


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_flux(a, b):
    fa, fb = 0.5*a*a, 0.5*b*b
    if a <= b:
        if a >= 0.0:
            return fa
        if b <= 0.0:
            return fb
        return 0.0
    return max(fa, fb)


def independently_reconstruct(input0, budget):
    """Scalar hand-style faces for EVERY step, not the tested vectorized flux."""
    u = input0.copy()
    checkpoint = {0: u.copy()}
    cumulative = 0.0
    tprev = 0.0
    max_step_error = 0.0
    for k, s in enumerate(budget, 1):
        require(int(s["step"]) == k, "nonconsecutive step index")
        start, end, dt = float(s["time_start"]), float(s["time_end"]), float(s["dt"])
        require(abs(start - tprev) <= 3e-14, "disconnected time budget")
        require(0.0 < dt <= DX*CFL + 1e-14, "invalid CFL/requested step cap")
        require(abs(dt / DX - float(s["cfl_actual"])) < 3e-13, "wrong actual CFL")
        require(abs((start + dt) - end) <= 3e-14, "wrong time endpoint")
        require(end <= TIMES[-1] + 3e-14, "overshoot terminal time")
        interfaces = np.empty(N, dtype=np.float64)
        for j in range(N):
            interfaces[j] = scalar_flux(float(u[j]), float(u[(j+1)%N]))
        fperiodic = scalar_flux(float(u[-1]), float(u[0]))
        require(abs(float(s["F_left"]) - fperiodic) <= 2e-14, "periodic left face differs")
        require(abs(float(s["F_right"]) - fperiodic) <= 2e-14, "periodic right face differs")
        before = DX * float(np.sum(u))
        nxt = np.empty_like(u)
        for j in range(N):
            nxt[j] = u[j] - dt/DX * (interfaces[j] - interfaces[(j-1)%N])
        after = DX * float(np.sum(nxt))
        require(abs(before - float(s["mass_before"])) < 2e-12, "mass_before mismatch")
        require(abs(after - float(s["mass_after"])) < 2e-12, "mass_after mismatch")
        require(abs(float(s["flux_budget_increment"])) < 1e-14, "periodic net flux should be zero")
        cumulative += dt * (fperiodic-fperiodic)
        require(abs(cumulative - float(s["flux_budget_cumulative"])) < 1e-14, "mass integral mismatch")
        require(abs(after - .5 - float(s["mass_defect"])) < 2e-12, "per-step budget mismatch")
        require(np.min(nxt) >= np.min(input0)-1e-12 and np.max(nxt) <= np.max(input0)+1e-12,
                "discrete max bound violated")
        require(np.isfinite(nxt).all(), "nonfinite numerical step")
        u = nxt
        tprev = end
        for time, count in ((0.3,75),(0.6,150),(0.7,175),(0.8,200)):
            if k == count:
                require(abs(end - time) < 2e-12, "step count/checkpoint mismatch")
                checkpoint[k] = u.copy()
        max_step_error = max(max_step_error, abs(float(s["mass_defect"])))
    require(len(budget) == 200, "expected exactly 200 positive steps")
    return checkpoint, max_step_error


def independent_exact_reference(time):
    """Independent physical-x 48-point Gauss average with scalar bisection."""
    z, w = np.polynomial.legendre.leggauss(48)
    assert time < T_BREAK
    output = np.empty(N)
    for i in range(N):
        x_left, x_right = i*DX, (i+1)*DX
        xq = .5*(x_left+x_right) + .5*DX*z
        if time == 0:
            xi = xq
        else:
            xi = np.empty_like(xq)
            for j, x in enumerate(xq):
                lower, upper = x-.75*time, x-.25*time
                for _ in range(64):
                    middle = .5*(lower+upper)
                    characteristic = middle + time*(.5+.25*math.sin(2*math.pi*middle))
                    if characteristic < x:
                        lower = middle
                    else:
                        upper = middle
                xi[j] = .5*(lower+upper)
        output[i] = .5 * float(np.dot(w, .5+.25*np.sin(2*math.pi*xi)))
    return output


def audit(output, config):
    d = Path(output)
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    require(cfg["periodic_case"]["times"] == list(TIMES), "changed times")
    require(cfg["periodic_case"]["N"] == N and cfg["base_requested_cfl"] == CFL, "changed mesh/CFL")
    info = json.loads((d/"run.json").read_text(encoding="utf-8"))
    require(info["experiment_id"] == "ET005-01" and info["status"] == "computed_unvalidated",
            "missing raw computation status")
    require(info["config_sha256"] == sha(config), "input config hash mismatch")
    for filename, expected_sha in info["output_sha256"].items():
        require(sha(d/filename) == expected_sha, "output hash mismatch: " + filename)
    require(set(info["output_sha256"]) == {"snapshots.csv", "metrics.csv", "step_budget.csv", "reference_cellavg.csv"}, "missing data")
    samples = rows(d/"snapshots.csv")
    references = rows(d/"reference_cellavg.csv")
    budget = rows(d/"step_budget.csv")
    metrics = rows(d/"metrics.csv")
    require(len(samples) == 5*N and len(references) == 3*N and len(metrics) == 5,
            "output cardinality mismatch")
    require([float(r["time"]) for r in metrics] == list(TIMES), "metric times mismatch")
    require([int(r["steps"]) for r in metrics] == [0,75,150,175,200], "step count mismatch")
    frames = {}
    for t in TIMES:
        subset = [r for r in samples if float(r["time"]) == t]
        require(len(subset) == N and [int(r["i"]) for r in subset] == list(range(N)),
                "missing or duplicate snapshot cells")
        for i, r in enumerate(subset):
            require(abs(float(r["x_left"]) - i*DX) <= 1e-14, "incorrect left face")
            require(abs(float(r["x_right"]) - (i+1)*DX) <= 1e-14, "incorrect right face")
        frames[t] = np.array([float(r["U"]) for r in subset])
    # Initial means derived from the exact sine antiderivative, independently.
    edges = np.arange(N+1)*DX
    exact_initial = .5 + (-np.cos(2*math.pi*edges[1:])+np.cos(2*math.pi*edges[:-1])) / (8*math.pi*DX)
    require(np.max(np.abs(frames[0] - exact_initial)) < 3e-14, "initial values are not exact cell means")
    recon, max_balance = independently_reconstruct(frames[0], budget)
    for time, k in ((0.0,0),(0.3,75),(0.6,150),(0.7,175),(0.8,200)):
        require(np.max(np.abs(frames[time] - recon[k])) <= 2e-13, "independent scalar FVM mismatch")
    previous_tv = None
    max_reference_discrepancy = 0.0
    for t, m in zip(TIMES, metrics):
        u = frames[t]
        mass = DX * float(np.sum(u))
        tv = float(np.abs(np.roll(u,-1)-u).sum())
        require(abs(mass-.5) < 1e-11 and abs(mass-float(m["mass"])) < 1e-12,
                "periodic mass/metric mismatch")
        require(abs(float(m["mass_defect"])) < 1e-11, "reported mass defect failed")
        require(abs(tv-float(m["total_variation"])) < 2e-12, "TV mismatch")
        require(m["TV_boundary"] == "periodic", "TV boundary type mismatch")
        require(np.min(u) >= np.min(frames[0])-1e-12 and np.max(u) <= np.max(frames[0])+1e-12,
                "maximum principle failed")
        if previous_tv is not None:
            require(tv <= previous_tv + 2e-12, "TV increased for monotone Godunov")
        previous_tv = tv
        if t < T_BREAK:
            subset = [r for r in references if float(r["time"]) == t]
            require(len(subset) == N and [int(r["i"]) for r in subset] == list(range(N)),
                    "missing smooth exact averages")
            exact = np.array([float(r["U_exact"]) for r in subset])
            separate = independent_exact_reference(t)
            max_reference_discrepancy = max(max_reference_discrepancy,
                                            float(np.max(np.abs(exact-separate))))
            require(max_reference_discrepancy < 2e-10, "independent characteristic reference mismatch")
            require(float(m["max_characteristic_residual"]) < 2e-13, "characteristic root residual")
            require(float(m["periodic_xi_shift_error"]) < 2e-13, "periodic lift inconsistent")
            L1 = DX * float(np.abs(u-exact).sum())
            require(abs(L1-float(m["L1_cellavg"])) < 2e-13, "cell-average L1 wrong")
            require(m["reference_kind"] == "smooth_exact_cellavg", "smooth reference status wrong")
            require(m["shock_theory_x"] == "" and m["shock_proxy_x"] == "", "early shock proxy present")
        else:
            require(m["reference_kind"] == "not_available_post_break", "postbreak reference dishonest")
            require(all(m[k] == "" for k in ("L1_cellavg", "max_characteristic_residual",
                  "max_reference_quadrature_disagreement", "periodic_xi_shift_error")),
                  "invalid post-break exact reference")
            theory = .5 + .5*t
            require(abs(float(m["shock_theory_x"])-theory) < 1e-14, "wrong theoretical shock trajectory")
            if m["shock_proxy_ambiguous"] == "False":
                jumps = np.roll(u,-1)-u
                j = int(np.argmin(jumps))
                require(abs(float(m["shock_proxy_x"])-(j+1)*DX) < 2e-14, "shock proxy wrong")
                err = (float(m["shock_proxy_x"])-theory+.5)%1.-.5
                require(abs(err-float(m["shock_proxy_error"]))<1e-14, "shock proxy error wrong")
            else:
                require(m["shock_proxy_ambiguous"] == "True", "invalid tie flag")
                require(m["shock_proxy_x"] == "" and m["shock_proxy_error"] == "", "tie hidden")
    require(max_balance <= 1e-11, "per-step conservation failed")
    return {"status": "PASS", "experiment_id": "ET005-01",
            "source_git_sha": info["source_git_sha"], "config_sha256": info["config_sha256"],
            "actual_steps": len(budget), "snapshots": len(samples), "reference_rows": len(references),
            "smooth_reference_times": list(REF_TIMES), "post_break_times": list(TIMES[3:]),
            "max_mass_defect": max_balance,
            "max_independent_reference_discrepancy": max_reference_discrepancy,
            "max_independent_FVM_discrepancy": max(float(np.max(abs(frames[t]-recon[k]))) for t,k in
                 ((0.0,0),(0.3,75),(0.6,150),(0.7,175),(0.8,200))),
            "snapshots_sha256": sha(d/"snapshots.csv"),
            "metrics_sha256": sha(d/"metrics.csv"),
            "step_budget_sha256": sha(d/"step_budget.csv"),
            "reference_cellavg_sha256": sha(d/"reference_cellavg.csv"),
            "validator_independent": "scalar_face_flux_plus_physical_gauss_48_point"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    target = args.out/"checks.json"
    try:
        result = audit(args.out, args.config)
    except Exception as exc:
        result = {"status": "FAIL", "experiment_id": "ET005-01",
                  "failed_reason": f"{type(exc).__name__}: {exc}"}
        target.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        print(result["failed_reason"], file=sys.stderr)
        return 1
    target.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
