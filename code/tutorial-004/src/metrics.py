"""Transparent metrics for the preregistered synthetic-image experiment."""
from __future__ import annotations

import math
import numpy as np

FLAT = (slice(65, 105), slice(50, 90))
EDGE = (slice(65, 105), slice(24, 36))
PROFILE_ROWS = slice(65, 105)
PROFILE_COLUMNS = np.arange(20, 46, dtype=int)
JUMP_DENOMINATOR = 0.78 - 0.20


def rmse(u: np.ndarray, reference: np.ndarray) -> float:
    delta = u - reference
    return float(np.sqrt(np.mean(delta * delta)))


def assess(u: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    if u.shape != (256, 256) or clean.shape != u.shape:
        raise ValueError("metrics expect the frozen 256x256 arrays")
    if not (np.isfinite(u).all() and np.isfinite(clean).all()):
        raise FloatingPointError("nonfinite pixel or reference")
    global_error = rmse(u, clean)
    local_flat = rmse(u[FLAT], clean[FLAT])
    local_edge = rmse(u[EDGE], clean[EDGE])
    jump = float(np.mean(u[PROFILE_ROWS, 30] - u[PROFILE_ROWS, 29]) / JUMP_DENOMINATOR)
    average = float(np.mean(u))
    centered_ss = float(np.sum((u - average) ** 2))
    return {
        "rmse": global_error,
        "psnr_db": math.inf if global_error == 0.0 else 20.0 * math.log10(1.0 / global_error),
        "rmse_flat": local_flat,
        "rmse_edge": local_edge,
        "edge_jump_ratio": jump,
        "mean": average,
        "minimum": float(np.min(u)),
        "maximum": float(np.max(u)),
        "centered_ss": centered_ss,
    }


def edge_profile(u: np.ndarray) -> np.ndarray:
    return np.mean(u[PROFILE_ROWS, 20:46], axis=0)
