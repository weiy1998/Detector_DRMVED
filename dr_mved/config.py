"""Configuration for the deterministic DR-MVED scoring map."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class DetectorConfig:
    """Hyperparameters fixed before a train/test evaluation.

    The defaults are deliberately conservative and dimension agnostic.  The
    Doppler bins default to the full FFT grid after construction of a block.
    """

    guard_radius: int = 2
    reference_radius: Optional[int] = None
    window_length: Optional[int] = None
    hop: int = 1
    n_fft: Optional[int] = None
    doppler_bins: Optional[Tuple[int, ...]] = None
    lambda_sparse: float = 0.025
    eta: float = 0.25
    proximal_stages: int = 3
    svd_energy: float = 0.65
    xi_q: float = 0.05
    xi_z: float = 0.05
    lambda_drift: float = 5.0
    ridge: float = 0.05
    beta: float = 0.50
    alpha: float = 0.02
    min_training_per_class: Optional[int] = None
    solver: str = "auto"

    def validate(self, slow_time: int, range_cells: int) -> None:
        if slow_time < 1 or range_cells < 3:
            raise ValueError("A block must have shape (M, N) with M>=1 and N>=3.")
        if self.guard_radius < 0:
            raise ValueError("guard_radius must be nonnegative.")
        if self.reference_radius is not None and self.reference_radius <= self.guard_radius:
            raise ValueError("reference_radius must exceed guard_radius.")
        if self.window_length is not None and not 1 <= self.window_length <= slow_time:
            raise ValueError("window_length must lie in [1, M].")
        if self.hop < 1:
            raise ValueError("hop must be positive.")
        if self.n_fft is not None and self.n_fft < 1:
            raise ValueError("n_fft must be positive.")
        if self.lambda_sparse <= 0 or self.eta <= 0:
            raise ValueError("lambda_sparse and eta must be positive.")
        if self.proximal_stages < 1:
            raise ValueError("proximal_stages must be at least one.")
        if not 0 < self.svd_energy <= 1:
            raise ValueError("svd_energy must lie in (0, 1].")
        if self.xi_q <= 0 or self.xi_z <= 0:
            raise ValueError("xi_q and xi_z must be positive.")
        if self.lambda_drift < 0 or self.ridge <= 0:
            raise ValueError("lambda_drift must be nonnegative and ridge positive.")
        # The scene-dependent lower bound beta >= 1/S is checked by the
        # projection fitter once the number of training scenes is known.
        if not 0 < self.beta <= 1:
            raise ValueError("beta must lie in (0, 1].")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        if self.doppler_bins is not None and len(self.doppler_bins) < 3:
            raise ValueError("At least three Doppler bins are required.")

    def with_dimensions(self, slow_time: int, n_fft: Optional[int] = None) -> "DetectorConfig":
        """Return a copy with an explicit FFT length if one was omitted."""

        chosen = n_fft if n_fft is not None else self.n_fft
        if chosen is None:
            chosen = slow_time
        return replace(self, n_fft=chosen)
