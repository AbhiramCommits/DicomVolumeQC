"""Patient-disjoint train/val/test splitting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from dicomvolumeqc.config import PreprocessConfig
from dicomvolumeqc.errors import ConfigError, SplitError
from dicomvolumeqc.models import SeriesMeta

SPLITS: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    """Whole-patient split assignments and achieved counts."""

    assignments: dict[str, str]
    patient_assignment: dict[str, str]
    patients_per_split: dict[str, int]
    series_per_split: dict[str, int]

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            split: {
                "patients": self.patients_per_split.get(split, 0),
                "series": self.series_per_split.get(split, 0),
            }
            for split in SPLITS
        }

    def summary(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "patients": dict(self.patient_assignment),
        }


def _validate_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLITS):
        raise ConfigError(f"split ratios must define exactly {SPLITS}")
    values = {split: float(ratios[split]) for split in SPLITS}
    if any(value < 0 for value in values.values()):
        raise ConfigError("split ratios must be non-negative")
    if abs(sum(values.values()) - 1.0) > 1e-6:
        raise ConfigError(f"split ratios must sum to 1.0, got {sum(values.values()):.6f}")
    return values


def patient_disjoint_split(
    series_metas: list[SeriesMeta],
    ratios: dict[str, float] | PreprocessConfig,
    seed: int,
) -> SplitResult:
    """Assign whole patients to splits, never splitting a patient across sets."""
    if isinstance(ratios, PreprocessConfig):
        ratios = ratios.split_ratio_map
    ratio_map = _validate_ratios(ratios)
    if not series_metas:
        raise SplitError("cannot split an empty list of series")

    by_patient: dict[str, list[SeriesMeta]] = {}
    seen_series: set[str] = set()
    for meta in series_metas:
        if not meta.patient_id:
            raise SplitError(f"series {meta.series_uid} has no patient identifier")
        if meta.series_uid in seen_series:
            raise SplitError(f"duplicate series identifier {meta.series_uid}")
        seen_series.add(meta.series_uid)
        by_patient.setdefault(meta.patient_id, []).append(meta)

    total = len(series_metas)
    remaining = {split: ratio_map[split] * total for split in SPLITS}
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    patient_assignment: dict[str, str] = {}
    for patient in patients:
        split = max(SPLITS, key=lambda name: (remaining[name], -SPLITS.index(name)))
        patient_assignment[patient] = split
        remaining[split] -= len(by_patient[patient])

    assignments: dict[str, str] = {}
    patients_per_split = {split: 0 for split in SPLITS}
    series_per_split = {split: 0 for split in SPLITS}
    for patient, split in patient_assignment.items():
        patients_per_split[split] += 1
        for meta in by_patient[patient]:
            assignments[meta.series_uid] = split
            series_per_split[split] += 1

    return SplitResult(
        assignments=assignments,
        patient_assignment=patient_assignment,
        patients_per_split=patients_per_split,
        series_per_split=series_per_split,
    )
