"""Deterministic synthetic DICOM CT corpus used by tests and demos.

Every series is a valid explicit-VR little-endian CT series with geometry
metadata, RescaleSlope/Intercept, and a binary NIfTI mask. Defects are injected
by explicit named flags so the QC layer can be tested against known ground
truth. No patient data is real.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

DEFAULT_SEED = 20240101
_UID_ROOT = "1.2.826.0.1.3680043.10.5432"
_AXIAL_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_IMPLEMENTATION_UID = f"{_UID_ROOT}.99"


@dataclass(frozen=True)
class StudyArtifacts:
    """Files and ground-truth flags for one generated series."""

    patient_id: str
    study_uid: str
    series_uid: str
    directory: Path
    files: tuple[Path, ...]
    mask_path: Path
    n_unique_slices: int
    defects: tuple[str, ...]


@dataclass(frozen=True)
class FixtureSpec:
    """Declarative definition of one series in the fixture corpus."""

    name: str
    patient_id: str
    n_slices: int = 12
    in_plane_spacing: tuple[float, float] = (0.8, 0.8)
    slice_spacing: float = 2.5
    defects: tuple[str, ...] = ()


def _deterministic_uid(*parts: object) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f"{_UID_ROOT}.{int(digest[:16], 16)}"


def _direction_from_orientation(orientation: tuple[float, ...]) -> tuple[float, ...]:
    row = np.asarray(orientation[:3], dtype=float)
    col = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row, col)
    return tuple(np.column_stack([row, col, normal]).ravel())


def _slice_normal(orientation: tuple[float, ...]) -> np.ndarray:
    return np.cross(np.asarray(orientation[:3]), np.asarray(orientation[3:]))


def _phantom(
    n_slices: int,
    rows: int,
    cols: int,
    spacing_xy: tuple[float, float],
    spacing_z: float,
    radius_mm: float,
    seed: int,
    phantom: str,
    extreme_hu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    x_mm = (xx - (cols - 1) / 2.0) * spacing_xy[0]
    y_mm = (yy - (rows - 1) / 2.0) * spacing_xy[1]
    z_mm = (np.arange(n_slices) - (n_slices - 1) / 2.0) * spacing_z

    sphere = (x_mm[None, :, :] ** 2 + y_mm[None, :, :] ** 2 + z_mm[:, None, None] ** 2) <= (
        radius_mm**2
    )
    if phantom == "cylinder":
        inside = (x_mm[None, :, :] ** 2 + y_mm[None, :, :] ** 2) <= radius_mm**2
        inside = np.broadcast_to(inside, sphere.shape).copy()
    else:
        inside = sphere

    hu = rng.normal(0.0, 25.0, size=sphere.shape).astype(np.float32)
    hu[inside] = 80.0 + rng.normal(0.0, 8.0, size=int(inside.sum())).astype(np.float32)
    if extreme_hu:
        hu[inside] = 31000.0
    mask = inside.astype(np.uint8)
    return np.clip(hu, -1000.0, 32000.0), mask


def make_study(
    dest: str | Path,
    *,
    patient_id: str = "PAT000",
    series_description: str = "synthetic",
    study_uid: str | None = None,
    series_uid: str | None = None,
    rows: int = 64,
    cols: int = 64,
    n_slices: int = 12,
    in_plane_spacing: tuple[float, float] = (0.8, 0.8),
    slice_spacing: float = 2.5,
    slice_thickness: float | None = None,
    orientation: tuple[float, ...] = _AXIAL_ORIENTATION,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius_mm: float = 16.0,
    seed: int = 0,
    phantom: str = "sphere",
    jitter_spacing: bool = False,
    drop_slice: bool = False,
    duplicate_slices: bool = False,
    shuffle_instance_numbers: bool = False,
    truncate: bool = False,
    single_slice: bool = False,
    phi: bool = False,
    extreme_hu: bool = False,
    mask_mismatch: bool = False,
    write_mask: bool = True,
) -> StudyArtifacts:
    """Write one synthetic CT series (and mask) to ``dest``.

    Flags inject exactly one defect each; the defects tuple records what was
    injected so tests can assert the QC verdict against the intended label.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    study_uid = study_uid or _deterministic_uid(patient_id, "study")
    series_uid = series_uid or _deterministic_uid(patient_id, series_description, seed)
    slice_thickness = float(slice_thickness if slice_thickness is not None else slice_spacing)

    normal = _slice_normal(orientation)
    positions = [
        np.asarray(origin, dtype=float) + normal * (k * slice_spacing) for k in range(n_slices)
    ]
    if single_slice:
        positions = positions[:1]
    if jitter_spacing:
        rng = np.random.default_rng(seed + 17)
        offsets = rng.uniform(-0.35, 0.35, size=len(positions)) * slice_spacing
        positions = [p + normal * o for p, o in zip(positions, offsets, strict=True)]
    positions.sort(key=lambda p: float(np.dot(p, normal)))
    if drop_slice and len(positions) > 3:
        positions.pop(len(positions) // 2)

    entries: list[tuple[np.ndarray, int]] = [(pos, k) for k, pos in enumerate(positions)]
    duplicate_source: int | None = None
    if duplicate_slices and entries:
        duplicate_source = len(entries) // 2
        entries.append(entries[duplicate_source])
        entries.append(entries[duplicate_source])

    n_written = len(entries)
    hu, mask = _phantom(
        n_written,
        rows,
        cols,
        in_plane_spacing,
        slice_spacing,
        radius_mm,
        seed,
        phantom,
        extreme_hu,
    )

    instance_numbers = list(range(1, n_written + 1))
    if shuffle_instance_numbers:
        instance_numbers = [int(v) for v in np.random.default_rng(seed + 3).permutation(
            np.arange(1, n_written + 1)
        )]
    if duplicate_source is not None:
        source_number = instance_numbers[duplicate_source]
        instance_numbers[-1] = source_number
        instance_numbers[-2] = source_number

    thickness_mm = slice_thickness
    defects: list[str] = []
    if jitter_spacing:
        defects.append("nonuniform_spacing")
    if drop_slice:
        defects.append("missing_slice")
    if duplicate_slices:
        defects.append("duplicate_slices")
    if orientation != _AXIAL_ORIENTATION:
        defects.append("tilted_orientation")
    if truncate:
        defects.append("truncated_file")
    if single_slice:
        defects.append("single_slice")
    if phi:
        defects.append("phi")
    if extreme_hu:
        defects.append("extreme_hu")
    if mask_mismatch:
        defects.append("mask_mismatch")

    files: list[Path] = []
    for idx, (position, source_idx) in enumerate(entries):
        sop_uid = _deterministic_uid(series_uid, "sop", idx)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = _IMPLEMENTATION_UID

        ds = FileDataset(str(dest / "placeholder"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientID = "MRN-884211" if phi else f"MRN-{patient_id}"
        ds.PatientName = "Doe^Jane" if phi else f"SYNTHETIC^{patient_id}"
        ds.PatientBirthDate = "19730102" if phi else "19800101"
        ds.PatientSex = "F" if phi else "O"
        ds.StudyDate = "20231115" if phi else "20240115"
        ds.SeriesDate = ds.StudyDate
        ds.InstitutionName = "Springfield General Hospital" if phi else "Synthetic Imaging Lab"
        ds.ReferringPhysicianName = "House^Gregory" if phi else ""
        ds.AccessionNumber = "ACC-99173" if phi else f"ACC-{patient_id}"
        ds.StudyID = "STUDY-1"
        ds.SeriesNumber = 1
        ds.SeriesDescription = series_description
        ds.InstanceNumber = instance_numbers[idx]
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelSpacing = [float(in_plane_spacing[0]), float(in_plane_spacing[1])]
        ds.SliceThickness = float(thickness_mm)
        ds.SpacingBetweenSlices = float(slice_spacing)
        ds.ImageOrientationPatient = [float(v) for v in orientation]
        ds.ImagePositionPatient = [float(v) for v in position]
        ds.SliceLocation = float(np.dot(position, normal))
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.WindowCenter = 40.0
        ds.WindowWidth = 400.0
        if phi:
            ds.add_new(0x00090010, "LO", "DVQC_PRIVATE_CREATOR")
            ds.add_new(0x00091001, "LO", "PRIVATE_MRN_884211")

        stored = np.rint(hu[source_idx] - ds.RescaleIntercept).astype(np.int16)
        ds.PixelData = stored.tobytes()
        path = dest / f"slice_{idx:03d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        files.append(path)

    if truncate and files:
        last = files[-1]
        payload = last.read_bytes()
        last.write_bytes(payload[: max(256, len(payload) - 512)])
        defects.append("unreadable_pixel_data")

    mask_path = dest / f"{dest.name}_mask.nii.gz"
    if write_mask:
        mask_spacing = (
            float(in_plane_spacing[0]) * (1.5 if mask_mismatch else 1.0),
            float(in_plane_spacing[1]) * (1.5 if mask_mismatch else 1.0),
            float(slice_spacing),
        )
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.SetSpacing(mask_spacing)
        mask_img.SetOrigin(tuple(float(v) for v in positions[0]))
        mask_img.SetDirection(_direction_from_orientation(orientation))
        sitk.WriteImage(mask_img, str(mask_path))

    return StudyArtifacts(
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=series_uid,
        directory=dest,
        files=tuple(files),
        mask_path=mask_path,
        n_unique_slices=len(positions),
        defects=tuple(dict.fromkeys(defects)),
    )


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


__all__ = ["DEFAULT_SEED", "FixtureSpec", "StudyArtifacts", "build_fixture_set", "make_study"]
