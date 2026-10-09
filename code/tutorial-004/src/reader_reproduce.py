"""Reader reproduction of ET004-03/04 without large private snapshot archives.

The original published numerical solver is imported, never duplicated. This
reader utility reruns only the preregistered finite sets of parameters and
compares the outputs with the officially frozen transparent CSV data. It does
not replace official scientific results or silently optimize any parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from diffusion_solver import advance_one_step
from image_inputs import make_synthetic_image, make_observation
from metrics import assess

ROOT = Path(__file__).resolve().parents[1]
CSV_SHA256 = {
    "ET004-02/metrics.csv": "e1c9e03813040f81231fa62a3bdabba790107b9fc2c572a6e4d142013b3b1413",
    "ET004-02/edge_profiles.csv": "04dd512ca82f4ff5d509cf2888623e5071477c24644ff57b3ca01378164578d0",
    "ET004-03/grid_metrics.csv": "d61df1a54bae332ac7e86e01cdd5439bac60b162b81e903fd2df38b63bb65fb9",
    "ET004-04/self_convergence.csv": "13cfca29d69611ee420d0321899d09b8c6f1876fe8ffcf8e819bf1e73041d7f9",
}
INPUT_SHA256 = {
    "clean": "5d7f522502e18c6c467d1661621bebbaf46fa9e435659df11e58992ad0068e8b",
    "noisy": "d1262e02d8cb9d9aa7f557b60560ea6cee340b61c7d8855bb4d7b270eb9aa987",
}


def freeze_rows(name: str) -> list[dict[str, str]]:
    if name not in CSV_SHA256:
        raise ValueError("unregistered frozen CSV: " + name)
    p = ROOT / "results" / name
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != CSV_SHA256[name]:
        raise AssertionError("frozen CSV identity changed: " + name)
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def inputs() -> tuple[np.ndarray, np.ndarray]:
    clean = make_synthetic_image()
    noisy = make_observation(clean)
    for key, u in (("clean", clean), ("noisy", noisy)):
        if u.dtype != np.float64 or u.shape != (256, 256):
            raise AssertionError("reader input must be float64 256x256")
        digest = hashlib.sha256(np.ascontiguousarray(u).tobytes(order="C")).hexdigest()
        if digest != INPUT_SHA256[key]:
            raise AssertionError("original " + key + " source pixel identity changed")
    return clean, noisy


def advance(u0: np.ndarray, *, dt: float, method: str, sigma: float = 1.5,
            kappa: float = 0.075, terminal: float = 16.0) -> np.ndarray:
    steps = round(terminal / dt)
    if dt <= 0 or abs(steps * dt - terminal) > 1e-12:
        raise ValueError("time samples must be exact dt multiples")
    u = u0.copy()
    for _ in range(steps):
        u = advance_one_step(u, dt=dt, h=1.0, method=method, sigma=sigma, kappa=kappa)
    return u


def compare(observed: float, frozen: float, label: str, tolerance: float = 1e-7) -> None:
    if not (math.isfinite(observed) and math.isfinite(frozen) and
            abs(observed - frozen) <= tolerance):
        raise AssertionError(f"{label}: observed {observed} vs frozen {frozen}")


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parameter_scan(clean: np.ndarray, noisy: np.ndarray, out: Path) -> dict:
    config = json.loads((ROOT / "config/ET004-03.json").read_text(encoding="utf-8"))
    if config["experiment_id"] != "ET004-03" or config["T"] != 16 or config["dt"] != 0.2:
        raise AssertionError("the nine-point scan is frozen")
    original = freeze_rows("ET004-03/grid_metrics.csv")
    if len(original) != 10:
        raise AssertionError("one heat baseline plus nine nonlinear samples required")
    heat = advance(noisy, dt=0.2, method="heat")
    compare(assess(heat, clean)["psnr_db"], float(original[0]["psnr_db"]), "heat baseline")
    rows = []
    for sigma in config["sigma_pixels"]:
        for kappa in config["kappa_intensity_per_pixel"]:
            found = [r for r in original if r["method"] == "regularized" and
                     float(r["sigma"]) == sigma and float(r["kappa"]) == kappa]
            if len(found) != 1:
                raise AssertionError("one row per preregistered parameter combination")
            u = advance(noisy, dt=0.2, method="regularized", sigma=sigma, kappa=kappa)
            psnr = assess(u, clean)["psnr_db"]
            reference = float(found[0]["psnr_db"])
            compare(psnr, reference, f"parameter {sigma},{kappa}")
            rows.append(dict(sigma=sigma, kappa=kappa, t=16,
                             reproduced_psnr_db=psnr, frozen_psnr_db=reference,
                             difference_db=psnr - reference))
    if len(rows) != 9:
        raise AssertionError("did not compute all nine pairs")
    write_rows(out / "reader_grid.csv", rows,
               ["sigma", "kappa", "t", "reproduced_psnr_db",
                "frozen_psnr_db", "difference_db"])
    return {"case": "ET004-03", "rows": 9,
            "maximum_psnr_difference_db": max(abs(r["difference_db"]) for r in rows)}


def time_refinement(clean: np.ndarray, noisy: np.ndarray, out: Path) -> dict:
    config = json.loads((ROOT / "config/ET004-04.json").read_text(encoding="utf-8"))
    dt = list(config["dt_levels"]) + [config["finer_reference_dt"]]
    if config["experiment_id"] != "ET004-04" or dt != [0.2, 0.1, 0.05, 0.025, 0.0125]:
        raise AssertionError("five levels of the preregistered time study changed")
    frozen = freeze_rows("ET004-04/self_convergence.csv")
    if len(frozen) != 5 or [float(row["dt"]) for row in frozen] != dt:
        raise AssertionError("frozen rows do not match the registered time levels")
    u = [advance(noisy, dt=h, method="regularized",
                 sigma=config["sigma_pixels"], kappa=config["kappa_intensity_per_pixel"])
         for h in dt]
    finer = u[-1]  # A numerical comparison state, NOT an exact continuous PDE solution.
    rows = []
    for h, state, reference in zip(dt, u, frozen):
        psnr = assess(state, clean)["psnr_db"]
        frozen_psnr = float(reference["psnr_db"])
        rmse_ref = float(np.sqrt(np.mean((state - finer) ** 2)))
        compare(psnr, frozen_psnr, f"PSNR dt={h}")
        compare(rmse_ref, float(reference["rmse_to_dt_0p0125_reference"]),
                f"time refinement dt={h}")
        rows.append(dict(dt=h, steps=round(16 / h), reproduced_psnr_db=psnr,
                         frozen_psnr_db=frozen_psnr, difference_db=psnr - frozen_psnr,
                         rmse_to_finer_numerical_state=rmse_ref))
    write_rows(out / "reader_time.csv", rows,
               ["dt", "steps", "reproduced_psnr_db", "frozen_psnr_db",
                "difference_db", "rmse_to_finer_numerical_state"])
    return {"case": "ET004-04", "rows": 5,
            "maximum_psnr_difference_db": max(abs(r["difference_db"]) for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered reader comparisons, not a new experiment")
    parser.add_argument("--case", choices=("check", "grid", "time", "both"), default="check")
    parser.add_argument("--out", type=Path, help="fresh directory; official frozen results never overwritten")
    args = parser.parse_args()

    for file in CSV_SHA256:
        freeze_rows(file)
    clean, noisy = inputs()

    if args.case == "check":
        print("READER_CSV_AND_SYNTHETIC_PIXEL_IDENTITY PASS; no PDE was run")
        return

    if args.out is None or args.out.exists():
        raise ValueError("pass --out pointing to a NEW directory; frozen CSVs are immutable")
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    if args.case in ("grid", "both"):
        records.append(parameter_scan(clean, noisy, args.out))
    if args.case in ("time", "both"):
        records.append(time_refinement(clean, noisy, args.out))
    (args.out / "reader_report.json").write_text(json.dumps(
        {"classification": "reader reproduction, NOT original canonical run",
         "comparisons": records}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2))
    print("READER_RECOMPUTE_WITH_REGISTERED_PARAMETERS PASS")


if __name__ == "__main__":
    main()
