"""Trainable DR-MVED detector and its frozen inference map."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .calibration import CalibrationBank
from .config import DetectorConfig
from .data import RadarSample
from .evidence import VIEW_NAMES, conditioned_evidence, doppler_profile, peak_excess
from .projection import ProjectionFailure, ProjectionFit, fit_projection
from .views import make_views


class DegenerateScaleError(ValueError):
    """Raised when a required training-derived scale is exactly degenerate."""


@dataclass
class DetectionResult:
    score: float
    p_value: float
    threshold: float
    detected: bool
    stratum: str
    rank_validity_available: bool


@dataclass
class _PreparedSample:
    sample: RadarSample
    profiles: Dict[str, np.ndarray]
    spectral_mads: Dict[str, np.ndarray]


class DRMVEDetector:
    """Domain-Regularized Multi-View Evidence Detector.

    ``fit`` freezes all training-derived quantities.  Subsequent calls to
    ``score`` and ``calibrate`` never use labels or alter the fitted map.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._fitted = False
        self.calibration_bank: CalibrationBank | None = None

    @staticmethod
    def _profiles(sample: RadarSample, config: DetectorConfig) -> _PreparedSample:
        config.validate(*sample.matrix.shape)
        views = make_views(sample.matrix, config)
        profiles: Dict[str, np.ndarray] = {}
        spectral_mads: Dict[str, np.ndarray] = {}
        for name in VIEW_NAMES:
            profile = doppler_profile(views[name], config)
            spectral_median = np.median(profile, axis=0)
            spectral_mad = np.median(np.abs(profile - spectral_median[None, :]), axis=0)
            profiles[name] = profile
            spectral_mads[name] = spectral_mad
        return _PreparedSample(sample, profiles, spectral_mads)

    @staticmethod
    def _q_profiles(prepared: _PreparedSample, epsilon_q: Mapping[str, float]) -> Dict[str, np.ndarray]:
        values = {}
        for name in VIEW_NAMES:
            q, _ = peak_excess(prepared.profiles[name], epsilon_q[name])
            values[name] = q
        return values

    def _feature_from_prepared(
        self,
        prepared: _PreparedSample,
        epsilon_q: Mapping[str, float],
        epsilon_z: Mapping[str, float],
    ) -> np.ndarray:
        q_profiles = self._q_profiles(prepared, epsilon_q)
        values = []
        for name in VIEW_NAMES:
            z, _, _, _ = conditioned_evidence(
                q_profiles[name],
                prepared.sample.cut_index,
                epsilon_z[name],
                self.config,
                prepared.sample.protected_indices,
            )
            values.append(z)
        return np.asarray(values, dtype=float)

    def fit(self, samples: Sequence[RadarSample]) -> "DRMVEDetector":
        """Fit scale floors and the domain-regularized projection."""

        samples = list(samples)
        if not samples:
            raise ValueError("At least one training sample is required.")
        if any(sample.label not in (0, 1) for sample in samples):
            raise ValueError("Training labels must be binary.")
        first_shape = samples[0].matrix.shape
        if any(sample.matrix.shape != first_shape for sample in samples):
            raise ValueError("All training blocks must have the same dimensions.")
        self.config.validate(*first_shape)
        prepared = [self._profiles(sample, self.config) for sample in samples]
        null_prepared = [item for item in prepared if item.sample.label == 0]
        if not null_prepared:
            raise ValueError("Training requires labelled null/clutter CUTs.")

        self.epsilon_q = {}
        for name in VIEW_NAMES:
            scales = np.concatenate([item.spectral_mads[name] for item in null_prepared])
            median_scale = float(np.median(scales))
            if median_scale <= 0 or not np.isfinite(median_scale):
                raise DegenerateScaleError(
                    f"Training spectral scale for view '{name}' is zero or nonfinite."
                )
            self.epsilon_q[name] = self.config.xi_q * median_scale

        local_scales: Dict[str, List[float]] = {name: [] for name in VIEW_NAMES}
        for item in null_prepared:
            q_profiles = self._q_profiles(item, self.epsilon_q)
            for name in VIEW_NAMES:
                _, _, scale, _ = conditioned_evidence(
                    q_profiles[name],
                    item.sample.cut_index,
                    0.0,
                    self.config,
                    item.sample.protected_indices,
                )
                local_scales[name].append(scale)
        self.epsilon_z = {}
        for name in VIEW_NAMES:
            median_scale = float(np.median(local_scales[name]))
            if median_scale <= 0 or not np.isfinite(median_scale):
                raise DegenerateScaleError(
                    f"Training local scale for view '{name}' is zero or nonfinite."
                )
            self.epsilon_z[name] = self.config.xi_z * median_scale

        features = np.vstack(
            [self._feature_from_prepared(item, self.epsilon_q, self.epsilon_z) for item in prepared]
        )
        labels = np.asarray([item.sample.label for item in prepared], dtype=int)
        scenes = np.asarray([item.sample.scene_id for item in prepared], dtype=object)
        self.projection = fit_projection(
            features,
            labels,
            scenes,
            lambda_drift=self.config.lambda_drift,
            ridge=self.config.ridge,
            beta=self.config.beta,
            solver=self.config.solver,
        )
        self.training_feature_dimension = int(features.shape[1])
        self.block_shape = tuple(int(x) for x in first_shape)
        self.training_margin = float(self.projection.kappa)
        self._fitted = True
        self.calibration_bank = None
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("The detector must be fitted before scoring.")

    def evidence(self, sample: RadarSample) -> np.ndarray:
        """Return the frozen three-view CUT-conditioned evidence vector."""

        self._require_fitted()
        if tuple(sample.matrix.shape) != self.block_shape:
            raise ValueError(f"Expected block shape {self.block_shape}, got {sample.matrix.shape}.")
        prepared = self._profiles(sample, self.config)
        return self._feature_from_prepared(prepared, self.epsilon_q, self.epsilon_z)

    def score(self, sample: RadarSample) -> float:
        """Return the final one-sided statistic ``w.T z``."""

        self._require_fitted()
        return float(self.projection.weight @ self.evidence(sample))

    def calibrate(
        self,
        null_samples: Sequence[RadarSample],
        alpha: float | None = None,
        split_valid: bool = False,
    ) -> CalibrationBank:
        """Fit a per-stratum empirical rank bank from null analysis units."""

        self._require_fitted()
        null_samples = list(null_samples)
        if not null_samples:
            raise ValueError("At least one null calibration sample is required.")
        scores: Dict[str, List[float]] = {}
        for sample in null_samples:
            if sample.label != 0:
                raise ValueError("Calibration samples must have label 0.")
            scores.setdefault(sample.stratum, []).append(self.score(sample))
        bank = CalibrationBank(split_valid=split_valid)
        bank.fit(scores, self.config.alpha if alpha is None else alpha)
        self.calibration_bank = bank
        return bank

    def predict(
        self, sample: RadarSample, alpha: float | None = None
    ) -> DetectionResult:
        """Score and, when calibrated, make the strict upper-tail decision."""

        score = self.score(sample)
        if self.calibration_bank is None:
            threshold = float("inf")
            p_value = 1.0
            detected = False
            valid = False
        else:
            threshold = self.calibration_bank.threshold(sample.stratum, alpha)
            p_value = self.calibration_bank.p_value(score, sample.stratum)
            detected = self.calibration_bank.decision(score, sample.stratum, alpha)
            valid = self.calibration_bank.formal_rank_validity_available
        return DetectionResult(score, p_value, threshold, detected, sample.stratum, valid)

    def save(self, path: str | Path) -> None:
        """Save all frozen parameters and optional calibration scores as JSON."""

        self._require_fitted()
        payload = {
            "config": _jsonable(asdict(self.config)),
            "epsilon_q": self.epsilon_q,
            "epsilon_z": self.epsilon_z,
            "block_shape": list(self.block_shape),
            "training_margin": self.training_margin,
            "projection": {
                "weight": self.projection.weight.tolist(),
                "scatter": self.projection.scatter.tolist(),
                "within_scatter": self.projection.within_scatter.tolist(),
                "drift_scatter": self.projection.drift_scatter.tolist(),
                "scene_deltas": self.projection.scene_deltas.tolist(),
                "mixture_weights": self.projection.mixture_weights.tolist(),
                "kappa": self.projection.kappa,
                "beta": self.projection.beta,
                "solver": self.projection.solver,
            },
            "calibration": None if self.calibration_bank is None else self.calibration_bank.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DRMVEDetector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config_payload = dict(payload["config"])
        if config_payload.get("doppler_bins") is not None:
            config_payload["doppler_bins"] = tuple(config_payload["doppler_bins"])
        detector = cls(DetectorConfig(**config_payload))
        projection_payload = payload["projection"]
        detector.epsilon_q = {key: float(value) for key, value in payload["epsilon_q"].items()}
        detector.epsilon_z = {key: float(value) for key, value in payload["epsilon_z"].items()}
        detector.block_shape = tuple(payload["block_shape"])
        detector.training_margin = float(payload["training_margin"])
        detector.training_feature_dimension = 3
        detector.projection = ProjectionFit(
            weight=np.asarray(projection_payload["weight"], dtype=float),
            scatter=np.asarray(projection_payload["scatter"], dtype=float),
            within_scatter=np.asarray(projection_payload["within_scatter"], dtype=float),
            drift_scatter=np.asarray(projection_payload["drift_scatter"], dtype=float),
            scene_deltas=np.asarray(projection_payload["scene_deltas"], dtype=float),
            mixture_weights=np.asarray(projection_payload["mixture_weights"], dtype=float),
            kappa=float(projection_payload["kappa"]),
            beta=float(projection_payload["beta"]),
            solver=str(projection_payload["solver"]),
        )
        detector._fitted = True
        if payload.get("calibration") is not None:
            detector.calibration_bank = CalibrationBank.from_dict(payload["calibration"])
        return detector


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
