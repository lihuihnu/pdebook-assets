#!/usr/bin/env python3
"""Execute the frozen tutorial-006 experiment; refuse any existing output path."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

from src.wave import (discrete_inner, displacement_l2, integer_step,
                      pulse, solve_wave)


ROOT = Path(__file__).resolve().parent
METRIC_FIELDS = (
    "run_id", "N", "boundary", "beta", "h", "dt", "step", "time",
    "l2_continuous_error", "linf_continuous_error",
    "l2_short_large_error", "linf_short_large_error",
    "l2_large_continuous_error", "linf_large_continuous_error",
    "l2_boundary_exact_error", "linf_boundary_exact_error",
    "reflection_target", "reflected_projection", "reflected_fit_l2",
    "reflected_template_l2",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def formatted(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return value


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow([formatted(value) for value in row])


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_identity() -> tuple[str | None, str]:
    candidate = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", candidate):
        return candidate.lower(), "GITHUB_SHA"
    try:
        candidate = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=10).strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", candidate):
            return candidate.lower(), "git"
    except (OSError, subprocess.SubprocessError):
        pass
    return None, "unavailable"


def validate_design(config: dict) -> None:
    """This executable implements one preregistered problem, not a tuner."""
    expected = {
        "model.c": (config["model"]["c"], 1),
        "model.short_domain": (config["model"]["short_domain"], [0, 1]),
        "model.large_domain": (config["model"]["large_domain"], [-2, 3]),
        "model.final_time": (config["model"]["final_time"], 1.2),
        "pulse.center": (config["model"]["pulse"]["center"], 0.35),
        "pulse.half_width": (config["model"]["pulse"]["half_width"], 0.1),
        "pulse.power": (config["model"]["pulse"]["power"], 6),
        "courant": (config["discretization"]["courant"], 0.8),
        "interval_counts": (config["discretization"]["interval_counts"],
                            [200, 400, 800, 1600]),
        "snapshot_times": (config["discretization"]["snapshot_times"],
                            [0, 0.4, 0.6, 0.9, 1.2]),
        "boundaries": (config["boundaries"], [
            {"name": "dirichlet", "kind": "dirichlet", "speed_ratio": None},
            {"name": "mur-beta-0p5", "kind": "mur", "speed_ratio": 0.5},
            {"name": "mur-beta-1", "kind": "mur", "speed_ratio": 1},
            {"name": "mur-beta-2", "kind": "mur", "speed_ratio": 2},
        ]),
    }
    if config.get("schema_version") != 1 or config.get("article_id") != "tutorial-006":
        raise ValueError("unsupported experiment identity or schema")
    for name, (actual, frozen) in expected.items():
        if actual != frozen:
            raise ValueError(f"{name} differs from the frozen experiment contract")
    for n in config["discretization"]["interval_counts"]:
        dt = config["discretization"]["courant"] / (n * config["model"]["c"])
        integer_step(config["model"]["final_time"], dt)
        for time in config["discretization"]["snapshot_times"]:
            integer_step(time, dt)


def save_run(out: Path, run_id: str, result, n: int, config: dict,
             *, kind: str, boundary: str, beta, domain) -> None:
    folder = out / "raw" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    write_csv(folder / "snapshots.csv", ("step", "time", "index", "x", "u"),
              ((step, step * result.dt, index, x, values[index])
               for step, values in sorted(result.snapshots.items())
               for index, x in enumerate(result.x)))
    write_csv(folder / "diagnostics.csv",
              ("step", "time", "min_u", "max_u", "l2_displacement"),
              result.diagnostics)
    cb_courant = None if beta is None else beta * config["discretization"]["courant"]
    q = None if cb_courant is None else (cb_courant - 1) / (cb_courant + 1)
    write_json(folder / "run.json", {
        "run_id": run_id, "N": n, "kind": kind, "boundary": boundary,
        "beta": beta, "c": config["model"]["c"], "h": result.h,
        "dt": result.dt, "courant": config["discretization"]["courant"],
        "boundary_courant": cb_courant, "q": q, "domain": list(domain),
        "crop": config["model"]["short_domain"], "crop_start": result.crop_start,
        "full_nodes": result.full_nodes,
        "final_time_requested": config["model"]["final_time"],
        "final_step": result.final_step,
        "snapshot_steps": sorted(result.snapshots),
        "diagnostics_scope": "full-domain", "snapshot_scope": "crop",
    })


def metric_rows(run_id: str, short, large, n: int, boundary: str,
                beta, config: dict):
    if not np.array_equal(short.x, large.x) or short.dt != large.dt:
        raise RuntimeError("short and large grids or fixed time steps differ")
    p = config["model"]["pulse"]
    c = config["model"]["c"]
    reflection_target = -1.0 if beta is None else (beta - 1.0) / (beta + 1.0)
    for step, values in sorted(short.snapshots.items()):
        time = step * short.dt
        reference = pulse(short.x - c * time, p["center"], p["half_width"], p["power"])
        reflected = pulse(2.0 - short.x - c * time,
                          p["center"], p["half_width"], p["power"])
        large_values = large.snapshots[step]
        continuous_error = values - reference
        short_large_error = values - large_values
        large_error = large_values - reference
        boundary_error = values - (reference + reflection_target * reflected)
        projection = fit = template_norm = None
        if step == short.final_step:
            denominator = discrete_inner(reflected, reflected, short.h)
            if denominator <= 0:
                raise RuntimeError("the final reflected template has zero norm")
            projection = discrete_inner(short_large_error, reflected, short.h) / denominator
            fit = displacement_l2(short_large_error - projection * reflected, short.h)
            template_norm = displacement_l2(reflected, short.h)
        yield (
            run_id, n, boundary, beta, short.h, short.dt, step, time,
            displacement_l2(continuous_error, short.h), float(np.max(np.abs(continuous_error))),
            displacement_l2(short_large_error, short.h), float(np.max(np.abs(short_large_error))),
            displacement_l2(large_error, short.h), float(np.max(np.abs(large_error))),
            displacement_l2(boundary_error, short.h), float(np.max(np.abs(boundary_error))),
            reflection_target, projection, fit, template_norm,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "design.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="new result directory; an existing path is always refused")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_design(config)
    out = args.out.resolve()
    # Atomic path creation guards completed and partial results alike.
    out.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    sources = {name: sha256(ROOT / name)
               for name in ("src/wave.py", "run_experiments.py")}
    commit, commit_source = source_identity()
    model = config["model"]
    discretization = config["discretization"]
    p = model["pulse"]
    shared = {
        "c": model["c"], "courant": discretization["courant"],
        "final_time": model["final_time"],
        "snapshot_times": discretization["snapshot_times"],
        "center": p["center"], "half_width": p["half_width"], "power": p["power"],
        "crop": model["short_domain"],
    }
    rows = []
    run_ids = []
    for n in discretization["interval_counts"]:
        large_id = f"large-N{n}"
        large = solve_wave(n, domain=model["large_domain"],
                           boundary="dirichlet", **shared)
        save_run(out, large_id, large, n, config, kind="large",
                 boundary="dirichlet", beta=None, domain=model["large_domain"])
        run_ids.append(large_id)
        for setting in config["boundaries"]:
            run_id = f"{setting['name']}-N{n}"
            beta = setting["speed_ratio"]
            short = solve_wave(n, domain=model["short_domain"],
                               boundary=setting["kind"],
                               boundary_speed_ratio=beta, **shared)
            save_run(out, run_id, short, n, config, kind="short",
                     boundary=setting["kind"], beta=beta, domain=model["short_domain"])
            rows.extend(metric_rows(run_id, short, large, n, setting["kind"],
                                    beta, config))
            run_ids.append(run_id)
            print(f"completed {run_id}: {short.final_step} fixed steps", flush=True)
    write_csv(out / "metrics.csv", METRIC_FIELDS, rows)
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("config changed during execution; results are incomplete")
    if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
        raise RuntimeError("source changed during execution; results are incomplete")
    artifacts = {path.relative_to(out).as_posix(): sha256(path)
                 for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "run.json", {
        "schema_version": 1, "article_id": "tutorial-006", "status": "completed",
        "config_path": str(config_path), "config_sha256": config_hash,
        "source_files": sources, "source_commit": commit,
        "source_commit_source": commit_source,
        "command": [sys.executable, *sys.argv],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "platform": platform.platform()},
        "started_utc": started, "completed_utc": utc_now(),
        "run_ids": run_ids, "artifact_sha256": artifacts,
    })
    print(f"wrote {len(run_ids)} runs and {len(rows)} metric rows to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
