"""Deterministic synthetic DICOM CT corpus used by tests and demos.

The low-level series writer lives in :mod:`dicomvolumeqc.fixture_study`; this
module declares the corpus and its intentionally broken variants.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dicomvolumeqc.fixture_study import DEFAULT_SEED, StudyArtifacts, make_study

_AXIAL_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class FixtureSpec:
    """Declarative definition of one series in the fixture corpus."""

    name: str
    patient_id: str
    n_slices: int = 12
    in_plane_spacing: tuple[float, float] = (0.8, 0.8)
    slice_spacing: float = 2.5
    defects: tuple[str, ...] = ()

_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec("clean_axial_070", "PAT001", 16, (0.70, 0.70)),
    FixtureSpec("clean_axial_080", "PAT002", 12, (0.80, 0.80)),
    FixtureSpec("clean_axial_060", "PAT003", 20, (0.60, 0.60)),
    FixtureSpec("clean_axial_100", "PAT005", 10, (1.00, 1.00)),
    FixtureSpec("clean_axial_125", "PAT006", 8, (1.25, 1.25)),
    FixtureSpec("clean_axial_090", "PAT007", 12, (0.90, 0.90)),
    FixtureSpec("clean_axial_075", "PAT004", 14, (0.75, 0.75)),
    FixtureSpec("clean_axial_065", "PAT002", 18, (0.65, 0.65)),
    FixtureSpec("nonuniform_spacing", "PAT004", 14, (0.80, 0.80), defects=("nonuniform_spacing",)),
    FixtureSpec("missing_slice", "PAT005", 12, (0.80, 0.80), defects=("missing_slice",)),
    FixtureSpec("duplicate_slices", "PAT006", 12, (0.80, 0.80), defects=("duplicate_slices",)),
    FixtureSpec("tilted_orientation", "PAT007", 12, (0.80, 0.80), defects=("tilted_orientation",)),
    FixtureSpec("phi_series", "PAT008", 12, (0.80, 0.80), defects=("phi",)),
    FixtureSpec("truncated_series", "PAT001", 8, (0.80, 0.80), defects=("truncated_file",)),
    FixtureSpec("single_slice", "PAT002", 1, (0.80, 0.80), defects=("single_slice",)),
    FixtureSpec("extreme_hu", "PAT003", 12, (0.80, 0.80), defects=("extreme_hu",)),
    FixtureSpec("mask_mismatch", "PAT004", 12, (0.80, 0.80), defects=("mask_mismatch",)),
)


def _make_from_spec(spec: FixtureSpec, dest: Path, seed: int) -> StudyArtifacts:
    defect = set(spec.defects)
    orientation = _AXIAL_ORIENTATION
    if "tilted_orientation" in defect:
        angle = np.deg2rad(20.0)
        orientation = (1.0, 0.0, 0.0, 0.0, float(np.cos(angle)), float(np.sin(angle)))
    n_slices = 1 if "single_slice" in defect else spec.n_slices
    return make_study(
        dest,
        patient_id=spec.patient_id,
        series_description=spec.name,
        n_slices=n_slices,
        in_plane_spacing=spec.in_plane_spacing,
        slice_spacing=spec.slice_spacing,
        orientation=orientation,
        seed=seed,
        phantom="sphere" if spec.name != "clean_axial_125" else "cylinder",
        jitter_spacing="nonuniform_spacing" in defect,
        drop_slice="missing_slice" in defect,
        duplicate_slices="duplicate_slices" in defect,
        shuffle_instance_numbers=spec.name in {"clean_axial_070", "clean_axial_100"},
        truncate="truncated_file" in defect,
        single_slice="single_slice" in defect,
        phi="phi" in defect,
        extreme_hu="extreme_hu" in defect,
        mask_mismatch="mask_mismatch" in defect,
    )


def build_fixture_set(root: str | Path, seed: int = DEFAULT_SEED) -> list[StudyArtifacts]:
    """Generate the complete deterministic corpus under ``root/raw``."""
    root = Path(root)
    raw = root / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)

    artifacts: list[StudyArtifacts] = []
    for index, spec in enumerate(_SPECS):
        dest = raw / spec.patient_id / spec.name
        artifacts.append(_make_from_spec(spec, dest, seed + index))

    manifest = {
        "seed": seed,
        "n_series": len(artifacts),
        "n_patients": len({a.patient_id for a in artifacts}),
        "series": [
            {
                "patient_id": a.patient_id,
                "series_uid": a.series_uid,
                "directory": str(a.directory.relative_to(root)),
                "files": len(a.files),
                "mask": str(a.mask_path.relative_to(root)),
                "defects": list(a.defects),
            }
            for a in artifacts
        ],
    }
    (root / "fixtures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return artifacts


__all__ = [
    "DEFAULT_SEED",
    "FixtureSpec",
    "StudyArtifacts",
    "build_fixture_set",
    "make_study",
]
