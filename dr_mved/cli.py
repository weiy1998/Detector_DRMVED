"""Command-line entry points for DR-MVED."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .data import load_npz, samples_from_arrays
from .detector import DRMVEDetector
from .synthetic import make_synthetic_dataset


def _demo(args: argparse.Namespace) -> int:
    config = DetectorConfig(alpha=args.alpha, beta=args.beta)
    dataset = make_synthetic_dataset(
        seed=args.seed,
        config=config,
        training_scenes=args.training_scenes,
        samples_per_class=args.samples_per_class,
        calibration_samples=args.calibration_samples,
        test_samples=args.test_samples,
    )
    detector = DRMVEDetector(config).fit(dataset.training)
    detector.calibrate(dataset.calibration, alpha=args.alpha, split_valid=True)
    results = [detector.predict(sample) for sample in dataset.test]
    positives = [result for result, sample in zip(results, dataset.test) if sample.label == 1]
    nulls = [result for result, sample in zip(results, dataset.test) if sample.label == 0]
    summary = {
        "training_margin": detector.training_margin,
        "projection_weight": detector.projection.weight.tolist(),
        "test_count": len(results),
        "target_detection_rate": float(np.mean([r.detected for r in positives])) if positives else None,
        "null_false_alarm_rate": float(np.mean([r.detected for r in nulls])) if nulls else None,
        "rank_validity_available": detector.calibration_bank.formal_rank_validity_available,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    detector.save(output / "dr_mved_model.json")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _fit_npz(args: argparse.Namespace) -> int:
    training_payload = load_npz(args.training)
    training = samples_from_arrays(**training_payload)
    detector = DRMVEDetector(DetectorConfig(alpha=args.alpha, beta=args.beta)).fit(training)
    if args.calibration:
        calibration_payload = load_npz(args.calibration)
        calibration = samples_from_arrays(**calibration_payload)
        detector.calibrate(calibration, alpha=args.alpha, split_valid=args.split_valid)
    detector.save(args.output)
    print(f"saved DR-MVED model to {args.output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dr-mved")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run a controlled end-to-end example")
    demo.add_argument("--output", default="dr_mved_demo")
    demo.add_argument("--seed", type=int, default=20260820)
    demo.add_argument("--alpha", type=float, default=0.02)
    demo.add_argument("--beta", type=float, default=0.50)
    demo.add_argument("--training-scenes", type=int, default=4)
    demo.add_argument("--samples-per-class", type=int, default=5)
    demo.add_argument("--calibration-samples", type=int, default=64)
    demo.add_argument("--test-samples", type=int, default=32)
    demo.set_defaults(handler=_demo)

    fit = subparsers.add_parser("fit-npz", help="fit from explicit NPZ arrays")
    fit.add_argument("--training", required=True)
    fit.add_argument("--calibration")
    fit.add_argument("--output", required=True)
    fit.add_argument("--alpha", type=float, default=0.02)
    fit.add_argument("--beta", type=float, default=0.50)
    fit.add_argument("--split-valid", action="store_true")
    fit.set_defaults(handler=_fit_npz)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
