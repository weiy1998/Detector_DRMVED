"""Small, label-aware evaluation helpers for held-out analysis units."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .data import RadarSample
from .detector import DRMVEDetector


@dataclass
class EvaluationSummary:
    count: int
    target_count: int
    null_count: int
    detection_probability: float
    empirical_false_alarm_probability: float
    mean_target_score: float
    mean_null_score: float
    mean_target_p_value: float
    rank_validity_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(detector: DRMVEDetector, samples: Sequence[RadarSample]) -> EvaluationSummary:
    """Evaluate target and null CUTs under one frozen model and bank."""

    samples = list(samples)
    results = [detector.predict(sample) for sample in samples]
    target_results = [result for result, sample in zip(results, samples) if sample.label == 1]
    null_results = [result for result, sample in zip(results, samples) if sample.label == 0]
    return EvaluationSummary(
        count=len(samples),
        target_count=len(target_results),
        null_count=len(null_results),
        detection_probability=float(np.mean([r.detected for r in target_results])) if target_results else float("nan"),
        empirical_false_alarm_probability=float(np.mean([r.detected for r in null_results])) if null_results else float("nan"),
        mean_target_score=float(np.mean([r.score for r in target_results])) if target_results else float("nan"),
        mean_null_score=float(np.mean([r.score for r in null_results])) if null_results else float("nan"),
        mean_target_p_value=float(np.mean([r.p_value for r in target_results])) if target_results else float("nan"),
        rank_validity_available=bool(detector.calibration_bank and detector.calibration_bank.formal_rank_validity_available),
    )

