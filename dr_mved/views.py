"""The three deterministic views used by DR-MVED."""

from __future__ import annotations

from typing import Dict

import numpy as np


def singular_value_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    left, values, right_h = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(values - threshold, 0.0)
    return (left * shrunk[None, :]) @ right_h


def group_soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    factors = np.maximum(1.0 - threshold / np.maximum(norms, 1e-15), 0.0)
    return matrix * factors[None, :]


def elementwise_soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    real = np.sign(matrix.real) * np.maximum(np.abs(matrix.real) - threshold, 0.0)
    imag = np.sign(matrix.imag) * np.maximum(np.abs(matrix.imag) - threshold, 0.0)
    return real + 1j * imag


def column_group_sparse_view(
    matrix: np.ndarray,
    lambda_sparse: float = 0.025,
    eta: float = 1.0,
    stages: int = 3,
) -> np.ndarray:
    """Return the finite-stage column-group sparse component ``O_K``."""

    if stages < 1 or lambda_sparse <= 0 or eta <= 0:
        raise ValueError("Invalid group-sparse proximal parameters.")
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    for _ in range(stages):
        low_rank = singular_value_threshold(matrix - sparse, 1.0 / eta)
        sparse = group_soft_threshold(matrix - low_rank, lambda_sparse / eta)
    return sparse


def elementwise_sparse_view(
    matrix: np.ndarray,
    lambda_sparse: float = 0.025,
    eta: float = 1.0,
    stages: int = 3,
) -> np.ndarray:
    """Reference ablation using independent real/imaginary shrinkage."""

    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    for _ in range(stages):
        low_rank = singular_value_threshold(matrix - sparse, 1.0 / eta)
        sparse = elementwise_soft_threshold(matrix - low_rank, lambda_sparse / eta)
    return sparse


def svd_residual_view(matrix: np.ndarray, energy_fraction: float = 0.65) -> np.ndarray:
    """Remove the smallest leading singular subspace reaching the energy cap."""

    if not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must lie in (0, 1].")
    left, values, right_h = np.linalg.svd(matrix, full_matrices=False)
    total = float(np.sum(values**2))
    if total <= 0:
        return matrix.copy()
    rank = int(np.searchsorted(np.cumsum(values**2), energy_fraction * total) + 1)
    dominant = (left[:, :rank] * values[None, :rank]) @ right_h[:rank, :]
    return matrix - dominant


def make_views(matrix: np.ndarray, config) -> Dict[str, np.ndarray]:
    """Construct raw, adaptive, and SVD-residual views."""

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    return {
        "raw": matrix,
        "adaptive": column_group_sparse_view(
            matrix, config.lambda_sparse, config.eta, config.proximal_stages
        ),
        "svd": svd_residual_view(matrix, config.svd_energy),
    }

