# DR-MVED

Reference implementation of the **Domain-Regularized Multi-View Evidence
Detector (DR-MVED)** described in the accompanying paper.

The implementation follows the paper's frozen score map:

1. Generate the raw, column-group-sparse, and deterministic SVD-residual views.
2. Extract a view-specific Doppler peak-excess evidence value.
3. Standardize the CUT with the median and MAD of its guarded secondary cells.
4. Learn a signed lower-tail projection with within-class scatter and
   cross-scene clutter-location drift regularization.
5. Calibrate the final score with null analysis units processed by the same
   map.

The package is intentionally model-driven and uses only NumPy and SciPy for
its required path. CVXPY is optional; the default solver computes the same
convex projection through the equivalent capped-simplex dual problem.

## Install

From this directory:

```bash
python -m pip install -r requirements.txt
```

The source tree can also be used directly with `PYTHONPATH=.`.

## Quick start

Run a small end-to-end controlled example:

```bash
python -m dr_mved.cli demo --output demo_output --seed 20260820
```

This writes the fitted model, calibration scores, and a JSON summary. The
demo is for implementation verification; it is not a replacement for the
paper's reported experiments.

## Python API

```python
from dr_mved import DRMVEDetector, DetectorConfig, RadarSample
from dr_mved.synthetic import make_synthetic_dataset

dataset = make_synthetic_dataset(seed=7, training_scenes=6)
detector = DRMVEDetector(DetectorConfig())
detector.fit(dataset.training)
detector.calibrate(dataset.calibration, alpha=0.02, split_valid=True)

result = detector.predict(dataset.test[0])
print(result.score, result.p_value, result.detected)
```

Each `RadarSample` contains one coherent pulse--range block, a CUT column,
the binary training label, a scene identifier, and a prespecified calibration
stratum. The block is retained as part of the analysis unit; surrounding
columns are never discarded before view generation.

For an annotated multi-cell target template, pass its zero-based cells as
`protected_indices`. The detector removes those cells and their configured
guard neighborhood from the local secondary set. When omitted, the secondary
set is the CUT-centered set defined only by `guard_radius` and
`reference_radius`.

## Input conventions

`RadarSample.matrix` must be a complex NumPy array of shape `(M, N)` where
`M` is slow time and `N` is range. `cut_index` is zero based. Training labels
are `0` for null/clutter CUTs and `1` for target CUTs. For measured radar data,
construct the samples from the same nonoverlapping blocks and metadata strata
used by the experimental protocol. The loader intentionally does not infer
target-free calibration units from annotations.

## Scope and validity

The finite-sample rank guarantee is exposed only as a conditional result when
the calibration bank is declared split-valid and its analysis units are
exchangeable with the test unit within a prespecified stratum. If a bank is
reused during fitting, the package still reports empirical p-values and
thresholds, but marks the formal rank-validity claim as unavailable.
