"""Data containers and a deliberately explicit NumPy input loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RadarSample:
    """One complete transductive radar analysis unit.

    Parameters
    ----------
    matrix:
        Complex pulse--range block with shape ``(M, N)``.
    cut_index:
        Zero-based range-cell index scored in this block.
    label:
        ``0`` for a null CUT and ``1`` for a target CUT.  Calibration samples
        should use ``0`` and are kept separate from projection fitting.
    scene_id:
        Training-scene identifier used by the lower-tail projection.
    stratum:
        Prespecified metadata stratum used for rank calibration.
    metadata:
        Optional acquisition metadata. It is carried through but never used to
        infer labels or merge strata.
    """

    matrix: np.ndarray
    cut_index: int
    label: int = 0
    scene_id: Any = 0
    stratum: str = "default"
    protected_indices: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix)
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be a two-dimensional array.")
        if not np.iscomplexobj(self.matrix):
            self.matrix = self.matrix.astype(np.complex128)
        if not 0 <= int(self.cut_index) < self.matrix.shape[1]:
            raise ValueError("cut_index is outside the range dimension.")
        if int(self.label) not in (0, 1):
            raise ValueError("label must be 0 or 1.")
        self.cut_index = int(self.cut_index)
        self.label = int(self.label)
        self.stratum = str(self.stratum)
        if self.protected_indices is not None:
            protected = tuple(sorted(set(int(index) for index in self.protected_indices)))
            if any(index < 0 or index >= self.matrix.shape[1] for index in protected):
                raise ValueError("protected_indices outside the range dimension.")
            if self.cut_index not in protected:
                protected = tuple(sorted(protected + (self.cut_index,)))
            self.protected_indices = protected


def samples_from_arrays(
    blocks: np.ndarray,
    labels: Sequence[int],
    cut_indices: Sequence[int] | int,
    scene_ids: Sequence[Any] | Any = 0,
    strata: Sequence[str] | str = "default",
) -> List[RadarSample]:
    """Create samples from ``(B, M, N)`` blocks and parallel metadata."""

    blocks = np.asarray(blocks)
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape (B, M, N).")
    count = blocks.shape[0]
    labels = list(labels)
    cuts = [cut_indices] * count if np.isscalar(cut_indices) else list(cut_indices)
    scenes = [scene_ids] * count if np.isscalar(scene_ids) else list(scene_ids)
    strata_list = [strata] * count if isinstance(strata, str) else list(strata)
    if not all(len(values) == count for values in (labels, cuts, scenes, strata_list)):
        raise ValueError("labels, cut_indices, scene_ids, and strata must have length B.")
    return [
        RadarSample(blocks[i], cuts[i], labels[i], scenes[i], strata_list[i])
        for i in range(count)
    ]


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a transparent NPZ dataset without guessing its semantics.

    Required key: ``blocks`` with shape ``(B, M, N)``. Optional keys are
    ``labels``, ``cut_indices``, ``scene_ids`` and ``strata``. Missing optional
    values default to null labels, the center range cell, scene ``0``, and the
    ``default`` stratum.
    """

    with np.load(Path(path), allow_pickle=True) as archive:
        if "blocks" not in archive:
            raise ValueError("NPZ input must contain a 'blocks' array.")
        blocks = archive["blocks"]
        count, _, n_range = blocks.shape
        labels = archive["labels"] if "labels" in archive else np.zeros(count, dtype=int)
        cuts = archive["cut_indices"] if "cut_indices" in archive else np.full(count, n_range // 2)
        scenes = archive["scene_ids"] if "scene_ids" in archive else np.zeros(count, dtype=int)
        strata = archive["strata"] if "strata" in archive else np.full(count, "default")
        return {
            "blocks": blocks,
            "labels": labels,
            "cut_indices": cuts,
            "scene_ids": scenes,
            "strata": strata,
        }
