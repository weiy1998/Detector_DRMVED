"""Domain-Regularized Multi-View Evidence Detector."""

from .config import DetectorConfig
from .data import RadarSample
from .detector import DRMVEDetector, DetectionResult

__all__ = ["DetectorConfig", "RadarSample", "DRMVEDetector", "DetectionResult"]

