"""Same-score-map empirical rank calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class CalibrationBank:
    """Per-stratum null score bank and finite-sample rank operations."""

    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    alpha: float = 0.02
    split_valid: bool = False

    def fit(self, scores: Mapping[str, Sequence[float]], alpha: float | None = None) -> None:
        if alpha is not None:
            self.alpha = float(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        self.scores = {}
        for stratum, values in scores.items():
            values = np.asarray(list(values), dtype=float)
            if values.ndim != 1 or np.any(~np.isfinite(values)):
                raise ValueError("Calibration scores must be finite one-dimensional arrays.")
            if values.size:
                self.scores[str(stratum)] = np.sort(values)

    def sample_count(self, stratum: str) -> int:
        return int(self.scores.get(str(stratum), np.empty(0)).size)

    def p_value(self, score: float, stratum: str = "default") -> float:
        values = self.scores.get(str(stratum))
        if values is None or values.size == 0:
            return 1.0
        return float((1 + np.count_nonzero(values >= score)) / (values.size + 1))

    def threshold(self, stratum: str = "default", alpha: float | None = None) -> float:
        alpha = self.alpha if alpha is None else float(alpha)
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1).")
        values = self.scores.get(str(stratum))
        if values is None or values.size == 0:
            return float("inf")
        k = int(np.ceil((values.size + 1) * (1.0 - alpha)))
        return float(values[k - 1]) if k <= values.size else float("inf")

    def decision(self, score: float, stratum: str = "default", alpha: float | None = None) -> bool:
        return bool(score > self.threshold(stratum, alpha))

    @property
    def formal_rank_validity_available(self) -> bool:
        return bool(self.split_valid)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "split_valid": self.split_valid,
            "scores": {key: values.tolist() for key, values in self.scores.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CalibrationBank":
        bank = cls(alpha=float(payload["alpha"]), split_valid=bool(payload["split_valid"]))
        bank.fit(payload.get("scores", {}))
        return bank

