"""Tests for DICOM discovery, 3D reconstruction, and geometry validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pydicom
import pytest

from dicomvolumeqc.errors import GeometryError
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series, load_volume, validate_geometry
from dicomvolumeqc.models import SeriesRef


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ingest_corpus")
    build_fixture_set(root)
    return root


@pytest.fixture(scope="module")
def refs(corpus: Path) -> list[SeriesRef]:
    return discover_series(corpus / "raw")


def _ref(refs: list[SeriesRef], name: str) -> SeriesRef:
    return next(ref for ref in refs if ref.directory.name == name)


def test_discovers_all_series_and_masks(refs: list[SeriesRef], corpus: Path) -> None:
    assert len(refs) == 17
    manifest = json.loads((corpus / "fixtures_manifest.json").read_text())
    assert {ref.series_uid for ref in refs} == {entry["series_uid"] for entry in manifest["series"]}
    assert all(ref.mask_path is not None for ref in refs)


@pytest.mark.parametrize(
    ("name", "shape", "spacing"),
    [
        ("clean_axial_070", (16, 64, 64), (0.70, 0.70)),
        ("clean_axial_080", (12, 64, 64), (0.80, 0.80)),
        ("clean_axial_060", (20, 64, 64), (0.60, 0.60)),
        ("clean_axial_100", (10, 64, 64), (1.00, 1.00)),
        ("clean_axial_125", (8, 64, 64), (1.25, 1.25)),
        ("clean_axial_090", (12, 64, 64), (0.90, 0.90)),
    ],
)
def test_clean_series_shapes_and_spacing(
    refs: list[SeriesRef], name: str, shape: tuple[int, int, int], spacing: tuple[float, float]
) -> None:
    volume = load_volume(_ref(refs, name))
    assert volume.shape == shape
    assert volume.meta.n_slices == shape[0]
    assert volume.meta.in_plane_spacing == spacing
    assert volume.spacing[0:2] == spacing
    assert volume.spacing[2] == pytest.approx(2.5, abs=1e-6)
    assert volume.meta.is_axial is True
    assert len(volume.meta.slice_positions) == shape[0]


def test_clean_series_hu_range(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "clean_axial_080"))
    assert -250.0 < volume.meta.intensity_min < 50.0
    assert 20.0 < volume.meta.intensity_max < 300.0
    assert volume.array.dtype.name == "float32"
    assert float(volume.array.max()) == pytest.approx(volume.meta.intensity_max)


def test_slice_ordering_uses_geometry_not_instance_number(refs: list[SeriesRef]) -> None:
    ref = _ref(refs, "clean_axial_070")
    pairs = []
    for path in ref.files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pairs.append((float(ds.ImagePositionPatient[2]), int(ds.InstanceNumber)))
    by_position = [inst for _, inst in sorted(pairs)]
    volume = load_volume(ref)
    assert list(volume.meta.slice_positions) == sorted(p for p, _ in pairs)
    assert list(volume.meta.instance_numbers) == by_position
    assert volume.meta.instance_numbers != tuple(sorted(volume.meta.instance_numbers))


def test_nonuniform_spacing_fixture(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "nonuniform_spacing"))
    codes = {f.code for f in validate_geometry(volume.meta)}
    assert "NONUNIFORM_SPACING" in codes
    assert volume.meta.slice_spacing_cv > 0.05


def test_missing_slice_fixture(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "missing_slice"))
    findings = validate_geometry(volume.meta)
    gap = next(f for f in findings if f.code == "SLICE_GAP")
    assert gap.severity.value == "FAIL"
    assert gap.observed == pytest.approx(5.0, abs=1e-3)


def test_duplicate_slices_detected_by_payload_hash(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "duplicate_slices"))
    assert volume.meta.duplicate_slices == 2
    assert volume.meta.n_slices == 14


def test_tilted_orientation_fixture(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "tilted_orientation"))
    findings = {f.code: f for f in validate_geometry(volume.meta)}
    assert "NON_AXIAL" in findings
    assert findings["NON_AXIAL"].observed == pytest.approx(20.0, abs=0.5)
    assert volume.meta.is_axial is False


def test_truncated_series_records_unreadable_file(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "truncated_series"))
    assert volume.meta.unreadable_count == 1
    assert volume.meta.file_count == 8
    assert volume.meta.n_slices == 7
    assert "CORRUPT_FILES" in {f.code for f in validate_geometry(volume.meta)}


def test_single_slice_fixture(refs: list[SeriesRef]) -> None:
    volume = load_volume(_ref(refs, "single_slice"))
    findings = validate_geometry(volume.meta)
    codes = {f.code for f in findings}
    assert "TOO_FEW_SLICES" in codes
    assert volume.spacing[2] == pytest.approx(2.5)


def test_clean_series_has_no_geometry_failures(refs: list[SeriesRef]) -> None:
    for name in ("clean_axial_070", "clean_axial_080", "clean_axial_060"):
        findings = validate_geometry(load_volume(_ref(refs, name)).meta)
        assert [f for f in findings if f.severity.value == "FAIL"] == []


def test_extreme_hu_and_mask_mismatch_are_loadable(refs: list[SeriesRef]) -> None:
    extreme = load_volume(_ref(refs, "extreme_hu"))
    assert extreme.meta.intensity_max > 30000
    mismatched = load_volume(_ref(refs, "mask_mismatch"))
    assert mismatched.meta.mask_path is not None


def test_inconsistent_geometry_finding(refs: list[SeriesRef]) -> None:
    meta = dataclasses.replace(
        load_volume(_ref(refs, "clean_axial_080")).meta, inconsistent_geometry=True
    )
    assert "INCONSISTENT_GEOMETRY" in {f.code for f in validate_geometry(meta)}


def test_non_orthonormal_orientation_finding(refs: list[SeriesRef]) -> None:
    meta = dataclasses.replace(
        load_volume(_ref(refs, "clean_axial_080")).meta,
        orientation_cosines=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    )
    assert "ORIENTATION_NOT_ORTHONORMAL" in {f.code for f in validate_geometry(meta)}


def test_empty_input_directory_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert discover_series(tmp_path / "empty") == []


def test_unreadable_file_is_tolerated(tmp_path: Path) -> None:
    (tmp_path / "junk.dcm").write_bytes(b"this is definitely not a DICOM file")
    refs = discover_series(tmp_path)
    assert len(refs) == 1
    assert refs[0].series_uid.startswith("UNREADABLE-")
    assert len(refs[0].unreadable_files) == 1
    with pytest.raises(GeometryError):
        load_volume(refs[0])


def test_load_volume_rejects_empty_series(tmp_path: Path) -> None:
    ref = SeriesRef(series_uid="EMPTY", directory=tmp_path)
    with pytest.raises(GeometryError, match="no readable slices"):
        load_volume(ref)


def test_spacing_cv_is_infinite_for_zero_median(refs: list[SeriesRef]) -> None:
    meta = dataclasses.replace(
        load_volume(_ref(refs, "clean_axial_080")).meta, slice_spacing_median=0.0
    )
    assert meta.slice_spacing_cv == float("inf")
    assert "NONUNIFORM_SPACING" not in {f.code for f in validate_geometry(meta)}
