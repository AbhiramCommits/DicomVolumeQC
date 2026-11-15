"""DICOM series discovery, 3D reconstruction, and geometry validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from dicomvolumeqc.errors import GeometryError
from dicomvolumeqc.geometry import validate_geometry
from dicomvolumeqc.models import SeriesMeta, SeriesRef, Volume

_SKIP_SUFFIXES = {
    ".nii",
    ".json",
    ".txt",
    ".csv",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".yaml",
    ".yml",
    ".npy",
    ".npz",
}


def _candidate_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name.lower().endswith(".nii.gz") or path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        yield path


def _mask_for(directory: Path) -> Path | None:
    masks = sorted(directory.glob("*.nii.gz"))
    return masks[0] if masks else None


def discover_series(root: str | Path) -> list[SeriesRef]:
    """Walk ``root`` and group readable DICOM files by SeriesInstanceUID.

    Unreadable files are retained on an explicit unreadable SeriesRef rather
    than crashing the run, so the QC layer can quarantine them.
    """
    root = Path(root)
    if not root.exists():
        return []
    groups: dict[str, dict[str, object]] = {}
    unreadable: dict[Path, list[Path]] = {}

    for path in _candidate_files(root):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            unreadable.setdefault(path.parent, []).append(path)
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not uid:
            unreadable.setdefault(path.parent, []).append(path)
            continue
        entry = groups.setdefault(
            uid,
            {
                "directory": path.parent,
                "files": [],
                "patient_id": str(getattr(ds, "PatientID", "") or ""),
                "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
            },
        )
        cast_files = entry["files"]
        assert isinstance(cast_files, list)
        cast_files.append(path)

    refs: list[SeriesRef] = []
    for uid, entry in groups.items():
        directory = entry["directory"]
        assert isinstance(directory, Path)
        files = tuple(sorted(entry["files"]))  # type: ignore[arg-type]
        refs.append(
            SeriesRef(
                series_uid=uid,
                directory=directory,
                files=files,
                patient_id=str(entry["patient_id"]),
                study_uid=str(entry["study_uid"]),
                mask_path=_mask_for(directory),
            )
        )
    for directory, files in sorted(unreadable.items()):
        refs.append(
            SeriesRef(
                series_uid=f"UNREADABLE-{directory.name}",
                directory=directory,
                files=(),
                unreadable_files=tuple(sorted(files)),
                mask_path=_mask_for(directory),
            )
        )
    return sorted(refs, key=lambda ref: (ref.series_uid, str(ref.directory)))


def _require(ds: Dataset, name: str, path: Path) -> object:
    value = getattr(ds, name, None)
    if value is None or value == "":
        raise GeometryError(f"missing {name} in {path}")
    return value


def _orientation(ds: Dataset, path: Path) -> np.ndarray:
    raw = _require(ds, "ImageOrientationPatient", path)
    values = np.asarray([float(v) for v in raw], dtype=float)  # type: ignore[union-attr]
    if values.shape != (6,):
        raise GeometryError(f"ImageOrientationPatient is not 6 values in {path}")
    return values


def _position(ds: Dataset, path: Path) -> np.ndarray:
    raw = _require(ds, "ImagePositionPatient", path)
    values = np.asarray([float(v) for v in raw], dtype=float)  # type: ignore[union-attr]
    if values.shape != (3,):
        raise GeometryError(f"ImagePositionPatient is not 3 values in {path}")
    return values


def load_volume(series_ref: SeriesRef) -> Volume:
    """Reconstruct a series into an HU volume sorted along the slice normal."""
    cache: list[tuple[Path, Dataset, np.ndarray]] = []
    unreadable = list(series_ref.unreadable_files)
    for path in series_ref.files:
        try:
            ds = pydicom.dcmread(path)
            pixels = ds.pixel_array
            if pixels.ndim != 2:
                raise GeometryError(f"pixel data is not 2D in {path}")
        except Exception:
            unreadable.append(path)
            continue
        cache.append((path, ds, pixels))

    if not cache:
        raise GeometryError(f"no readable slices in series {series_ref.series_uid}")

    first_ds = cache[0][1]
    try:
        orientation = _orientation(first_ds, cache[0][0])
        row_cosines = orientation[:3]
        col_cosines = orientation[3:]
        normal = np.cross(row_cosines, col_cosines)
        norm = float(np.linalg.norm(normal))
        if norm == 0:
            raise GeometryError("degenerate ImageOrientationPatient")
        normal = normal / norm
    except Exception as exc:
        raise GeometryError(f"unusable geometry in series {series_ref.series_uid}: {exc}") from exc

    def project(item: tuple[Path, Dataset, np.ndarray]) -> float:
        return float(np.dot(_position(item[1], item[0]), normal))

    try:
        ordered = sorted(cache, key=project)
    except Exception as exc:
        raise GeometryError(f"unusable slice positions in {series_ref.series_uid}: {exc}") from exc

    rows_cols = {(int(ds.Rows), int(ds.Columns)) for _, ds, _ in ordered}
    pixel_spacings = {
        (round(float(ds.PixelSpacing[0]), 6), round(float(ds.PixelSpacing[1]), 6))
        for _, ds, _ in ordered
    }
    orientations = {
        tuple(round(float(v), 6) for v in _orientation(ds, path)) for path, ds, _ in ordered
    }
    inconsistent = len(rows_cols) > 1 or len(pixel_spacings) > 1 or len(orientations) > 1

    shapes = {arr.shape for _, _, arr in ordered}
    if len(shapes) > 1:
        raise GeometryError(f"inconsistent Rows/Columns in series {series_ref.series_uid}")

    hu_slices = []
    hashes = []
    for _, ds, pixels in ordered:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_slices.append(pixels.astype(np.float32) * slope + intercept)
        hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
    volume = np.stack(hu_slices).astype(np.float32, copy=False)

    positions = np.asarray([project(item) for item in ordered], dtype=float)
    diffs = np.diff(positions)
    positive = diffs[diffs > 1e-4]
    if positive.size:
        spacing_z = float(np.median(positive))
        spacing_z_std = float(np.std(positive)) if positive.size > 1 else 0.0
    else:
        spacing_z = float(getattr(first_ds, "SliceThickness", 1.0) or 1.0)
        spacing_z_std = 0.0

    px = float(first_ds.PixelSpacing[1])
    py = float(first_ds.PixelSpacing[0])
    axial_angle = float(
        np.degrees(np.arccos(np.clip(abs(float(normal[2])), -1.0, 1.0)))
    )
    meta = SeriesMeta(
        patient_id=series_ref.patient_id
        or str(getattr(first_ds, "PatientID", "") or ""),
        study_uid=series_ref.study_uid
        or str(getattr(first_ds, "StudyInstanceUID", "") or ""),
        series_uid=series_ref.series_uid,
        modality=str(getattr(first_ds, "Modality", "") or ""),
        n_slices=len(ordered),
        in_plane_spacing=(px, py),
        slice_spacing_median=spacing_z,
        slice_spacing_std=spacing_z_std,
        orientation_cosines=tuple(float(v) for v in orientation),
        is_axial=axial_angle <= 5.0,
        volume_shape=tuple(int(v) for v in volume.shape),  # type: ignore[arg-type]
        intensity_min=float(volume.min()),
        intensity_max=float(volume.max()),
        file_count=len(series_ref.files) + len(series_ref.unreadable_files),
        unreadable_count=len(unreadable),
        slice_positions=tuple(float(v) for v in positions),
        instance_numbers=tuple(int(getattr(ds, "InstanceNumber", 0) or 0) for _, ds, _ in ordered),
        duplicate_slices=len(hashes) - len(set(hashes)),
        inconsistent_geometry=inconsistent,
        constant_volume=bool(float(np.ptp(volume)) < 1e-6),
        mask_path=str(series_ref.mask_path) if series_ref.mask_path else None,
        directory=str(series_ref.directory),
        series_description=str(getattr(first_ds, "SeriesDescription", "") or ""),
    )
    direction = tuple(
        float(v)
        for v in np.column_stack([row_cosines, col_cosines, normal]).ravel()
    )
    origin = tuple(float(v) for v in _position(ordered[0][1], ordered[0][0]))
    return Volume(
        array=volume,
        spacing=(px, py, spacing_z),
        origin=origin,
        direction=direction,
        meta=meta,
    )


__all__ = ["discover_series", "load_volume", "validate_geometry"]
