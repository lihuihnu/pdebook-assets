"""Deterministic ET004 synthetic reference and noisy observation.

A pixel is U[j, i], with i horizontal and j vertical.
The reference geometry and noise stream are preregistered, not optimized.
"""
from __future__ import annotations

import hashlib
import numpy as np


def input_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    if a.dtype != np.float64 or a.ndim != 2:
        raise TypeError("hash input must be float64 2D")
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def make_synthetic_image(N: int = 256) -> np.ndarray:
    if N != 256:
        raise ValueError("ET004-02 freezes the reference image to 256x256")
    image = np.full((N, N), 0.20, dtype=np.float64)
    image[40:125, 30:110] = 0.78
    j, i = np.ogrid[:N, :N]
    image[(i - 181) ** 2 + (j - 174) ** 2 <= 35 ** 2] = 0.58
    gradient = 0.25 + 0.45 * (np.arange(150, 246) - 150) / 95.0
    image[25:90, 150:246] = gradient[None, :]
    return image


def make_observation(
    clean: np.ndarray, *, seed: int = 20261008, noise_std: float = 0.08
) -> np.ndarray:
    if seed != 20261008 or noise_std != 0.08 or clean.shape != (256, 256):
        raise ValueError("ET004-02 frozen noise contract has changed")
    generator = np.random.Generator(np.random.PCG64(seed))
    noise = generator.normal(loc=0.0, scale=noise_std, size=clean.shape)
    return clean + noise
