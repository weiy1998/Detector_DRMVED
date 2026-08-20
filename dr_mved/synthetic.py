"""Small controlled compound-Gaussian maritime radar generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import DetectorConfig
from .data import RadarSample


@dataclass
class SyntheticDataset:
    training: List[RadarSample]
    calibration: List[RadarSample]
    test: List[RadarSample]


def _correlation_cholesky(slow_time: int, rho: float = 0.92, doppler: float = 0.06) -> np.ndarray:
    indices = np.arange(slow_time)
    difference = indices[:, None] - indices[None, :]
    covariance = rho ** np.abs(difference)
    covariance = covariance * np.exp(1j * 2 * np.pi * doppler * difference)
    return np.linalg.cholesky(covariance + 1e-8 * np.eye(slow_time))


def generate_block(
    rng: np.random.Generator,
    config: DetectorConfig,
    cholesky: np.ndarray,
    p_scale: float,
    power_drift: float,
    target_present: bool,
    scr_db: float = 8.0,
    target_width: int = 2,
    texture_shape: float = 1.5,
    target_doppler: float = 0.18,
    n_range: int = 20,
) -> np.ndarray:
    """Generate one complex pulse--range block."""

    slow_time = cholesky.shape[0]
    if n_range < 3:
        raise ValueError("n_range must be at least three.")
    column_indices = np.arange(n_range)
    center = n_range // 2
    power = p_scale * np.exp(power_drift * (column_indices - center) / max(n_range, 1))
    gaussian = (
        rng.normal(size=(slow_time, n_range))
        + 1j * rng.normal(size=(slow_time, n_range))
    ) / np.sqrt(2.0)
    texture = rng.gamma(texture_shape, 1.0 / texture_shape, size=n_range)
    clutter = cholesky @ gaussian
    clutter = clutter * np.sqrt(power * texture)[None, :]
    if not target_present:
        return clutter

    start = max(0, center - target_width // 2)
    stop = min(n_range, start + target_width)
    range_profile = np.zeros(n_range)
    range_profile[start:stop] = 1.0 / np.sqrt(max(1, stop - start))
    samples = np.arange(slow_time)
    target = np.zeros((slow_time, n_range), dtype=complex)
    duration_start = max(0, (slow_time - min(16, slow_time)) // 2)
    duration_stop = min(slow_time, duration_start + min(16, slow_time))
    target[duration_start:duration_stop, :] = (
        np.exp(1j * (2 * np.pi * target_doppler * samples[duration_start:duration_stop, None]))
        * range_profile[None, :]
    )
    clutter_energy = float(np.sum(power[start:stop]) * np.trace(cholesky @ cholesky.conj().T).real)
    target_energy = float(np.sum(np.abs(target) ** 2))
    amplitude = np.sqrt(10 ** (scr_db / 10.0) * clutter_energy / max(target_energy, 1e-12))
    phase = rng.uniform(0.0, 2 * np.pi)
    return clutter + amplitude * np.exp(1j * phase) * target


def make_synthetic_dataset(
    seed: int = 7,
    config: DetectorConfig | None = None,
    training_scenes: int = 4,
    samples_per_class: int = 5,
    calibration_samples: int = 64,
    test_samples: int = 32,
) -> SyntheticDataset:
    """Return a deterministic dataset suitable for an end-to-end smoke run."""

    config = config or DetectorConfig()
    slow_time = 24 if config.window_length is None else max(config.window_length, 16)
    n_range = 20
    cholesky = _correlation_cholesky(slow_time)
    rng = np.random.default_rng(seed)
    training: List[RadarSample] = []
    for scene in range(training_scenes):
        p_scale = float(np.exp(rng.normal(0.0, 0.30)))
        drift = float(rng.normal(0.0, 0.80))
        for label in (0, 1):
            for _ in range(samples_per_class):
                matrix = generate_block(
                    rng,
                    config,
                    cholesky,
                    p_scale,
                    drift,
                    bool(label),
                    scr_db=8.0,
                    n_range=n_range,
                )
                training.append(
                    RadarSample(
                        matrix,
                        n_range // 2,
                        label,
                        scene,
                        "synthetic",
                        tuple(range(n_range // 2 - 1, n_range // 2 + 1)),
                    )
                )
    calibration = []
    for _ in range(calibration_samples):
        matrix = generate_block(rng, config, cholesky, 1.0, 0.0, False, n_range=n_range)
        calibration.append(
            RadarSample(
                matrix,
                n_range // 2,
                0,
                0,
                "synthetic",
                tuple(range(n_range // 2 - 1, n_range // 2 + 1)),
            )
        )
    test = []
    for index in range(test_samples):
        matrix = generate_block(
            rng,
            config,
            cholesky,
            1.8,
            1.5,
            bool(index % 2),
            scr_db=8.0,
            n_range=n_range,
        )
        test.append(
            RadarSample(
                matrix,
                n_range // 2,
                index % 2,
                "heldout",
                "synthetic",
                tuple(range(n_range // 2 - 1, n_range // 2 + 1)),
            )
        )
    return SyntheticDataset(training, calibration, test)
