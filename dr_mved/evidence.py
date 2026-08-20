"""Doppler evidence and CUT-conditioned robust standardization."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


VIEW_NAMES = ("raw", "adaptive", "svd")
MAD_NORMALIZER = 1.4826


def reference_indices(
    n_range: int,
    cut_index: int,
    guard_radius: int,
    reference_radius: int | None = None,
    protected_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return ``{j: G < |j-i| <= R}`` with no circular padding."""

    if not 0 <= cut_index < n_range:
        raise ValueError("cut_index outside range dimension.")
    indices = np.arange(n_range)
    distance = np.abs(indices - cut_index)
    mask = distance > guard_radius
    if protected_indices is not None:
        protected = np.asarray(list(protected_indices), dtype=int)
        if protected.size:
            protected_distance = np.min(np.abs(indices[:, None] - protected[None, :]), axis=1)
            mask &= protected_distance > guard_radius
    if reference_radius is not None:
        mask &= distance <= reference_radius
    result = indices[mask]
    if result.size == 0:
        raise ValueError("The CUT has no eligible secondary cells.")
    return result


def _hann_window(length: int) -> np.ndarray:
    if length == 1:
        return np.ones(1, dtype=float)
    window = np.hanning(length)
    norm = np.linalg.norm(window)
    return window / norm if norm > 0 else np.ones(length, dtype=float) / np.sqrt(length)


def doppler_profile(view: np.ndarray, config) -> np.ndarray:
    """Compute the averaged windowed Doppler power profile ``(Nf, N)``."""

    slow_time, n_range = view.shape
    length = slow_time if config.window_length is None else config.window_length
    n_fft = slow_time if config.n_fft is None else config.n_fft
    if not 1 <= length <= slow_time:
        raise ValueError("window_length must lie in [1, M].")
    window = _hann_window(length)
    starts = range(0, slow_time - length + 1, config.hop)
    frames = []
    for start in starts:
        frame = view[start : start + length, :] * window[:, None]
        spectrum = np.fft.fftshift(np.fft.fft(frame, n=n_fft, axis=0), axes=0)
        frames.append(np.abs(spectrum) ** 2)
    if not frames:
        raise ValueError("No complete Doppler frames are available.")
    profile = np.mean(np.stack(frames, axis=0), axis=0)
    bins = config.doppler_bins
    if bins is not None:
        bins = np.asarray(bins, dtype=int)
        if np.any(bins < 0) or np.any(bins >= n_fft):
            raise ValueError("doppler_bins contains an invalid FFT index.")
        profile = profile[bins, :]
    if profile.shape[0] < 3:
        raise ValueError("At least three Doppler bins are required.")
    return profile


def peak_excess(profile: np.ndarray, epsilon_q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return one evidence value per range cell and its spectral MAD."""

    spectral_median = np.median(profile, axis=0)
    spectral_mad = np.median(np.abs(profile - spectral_median[None, :]), axis=0)
    q = (np.max(profile, axis=0) - spectral_median) / (spectral_mad + epsilon_q)
    return q.astype(float), spectral_mad.astype(float)


def local_location_scale(q: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(q)[reference]
    location = float(np.median(values))
    scale = float(MAD_NORMALIZER * np.median(np.abs(values - location)))
    return location, scale


def conditioned_evidence(
    q: np.ndarray,
    cut_index: int,
    epsilon_z: float,
    config,
    protected_indices: Iterable[int] | None = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Return ``(z, local_location, local_scale, reference_indices)``."""

    reference = reference_indices(
        q.size,
        cut_index,
        config.guard_radius,
        config.reference_radius,
        protected_indices,
    )
    location, scale = local_location_scale(q, reference)
    z = (float(q[cut_index]) - location) / (scale + epsilon_z)
    return float(z), location, scale, reference


def feature_vector(
    views: Dict[str, np.ndarray],
    cut_index: int,
    epsilon_q: Dict[str, float],
    epsilon_z: Dict[str, float],
    config,
    protected_indices: Iterable[int] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute the three-dimensional CUT evidence vector.

    The second return value contains per-view ``q`` profiles and the third
    contains per-view spectral MAD arrays, which are useful while fitting the
    training floors.
    """

    q_profiles: Dict[str, np.ndarray] = {}
    spectral_mads: Dict[str, np.ndarray] = {}
    values = []
    for name in VIEW_NAMES:
        profile = doppler_profile(views[name], config)
        q, spectral_mad = peak_excess(profile, epsilon_q[name])
        q_profiles[name] = q
        spectral_mads[name] = spectral_mad
        z, _, _, _ = conditioned_evidence(
            q, cut_index, epsilon_z[name], config, protected_indices
        )
        values.append(z)
    return np.asarray(values, dtype=float), q_profiles, spectral_mads


def mean_std_conditioned_evidence(
    q: np.ndarray, cut_index: int, epsilon_z: float, config
) -> float:
    """Mean/std scaling used only as the paper's robustness ablation."""

    reference = reference_indices(
        q.size, cut_index, config.guard_radius, config.reference_radius
    )
    values = q[reference]
    center = float(np.mean(values))
    scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float((q[cut_index] - center) / (scale + epsilon_z))
