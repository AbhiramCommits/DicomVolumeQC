"""Tests for the deterministic synthetic DICOM fixture generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

from dicomvolumeqc.fixtures import _SPECS, build_fixture_set


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fixture_corpus")
    build_fixture_set(root)
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / "fixtures_manifest.json").read_text())


def test_corpus_size_and_patient_count(corpus: Path) -> None:
    manifest = _manifest(corpus)
    assert manifest["n_series"] == len(_SPECS) == 17
    assert manifest["n_patients"] == 8
    assert manifest["n_series"] >= 12
    assert manifest["n_patients"] >= 8


def test_every_series_directory_contains_dicom_and_mask(corpus: Path) -> None:
    for entry in _manifest(corpus)["series"]:
        directory = corpus / entry["directory"]
        assert len(list(directory.glob("*.dcm"))) == entry["files"]
        assert (corpus / entry["mask"]).is_file()


def test_clean_series_is_readable_with_pydicom(corpus: Path) -> None:
    entry = next(e for e in _manifest(corpus)["series"] if "clean_axial_080" in e["directory"])
    files = sorted((corpus / entry["directory"]).glob("*.dcm"))
    ds = pydicom.dcmread(files[0])
    assert ds.Modality == "CT"
    assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2"
    assert (ds.Rows, ds.Columns) == (64, 64)
    assert ds.PixelSpacing == [0.80, 0.80]
    assert ds.pixel_array.dtype.name == "int16"
    assert ds.pixel_array.shape == (64, 64)
    assert ds.ImageOrientationPatient == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert len(ds.StudyInstanceUID) <= 64


def test_all_expected_defects_are_present(corpus: Path) -> None:
    defects = {d for entry in _manifest(corpus)["series"] for d in entry["defects"]}
    assert {
        "nonuniform_spacing",
        "missing_slice",
        "duplicate_slices",
        "tilted_orientation",
        "truncated_file",
        "single_slice",
        "phi",
        "extreme_hu",
        "mask_mismatch",
    } <= defects


def test_truncated_series_has_an_unreadable_pixel_payload(corpus: Path) -> None:
    entry = next(e for e in _manifest(corpus)["series"] if "truncated_series" in e["directory"])
    files = sorted((corpus / entry["directory"]).glob("*.dcm"))
    ds = pydicom.dcmread(files[-1])
    with pytest.raises(ValueError):
        _ = ds.pixel_array
    header = pydicom.dcmread(files[-1], stop_before_pixels=True, force=True)
    assert header.Rows == 64


def test_masks_are_binary_and_spatially_close(corpus: Path) -> None:
    entry = next(e for e in _manifest(corpus)["series"] if "clean_axial_080" in e["directory"])
    mask = sitk.ReadImage(str(corpus / entry["mask"]))
    values = sitk.GetArrayFromImage(mask)
    assert set(int(v) for v in values.ravel()) <= {0, 1}
    assert values.ndim == 3
    assert values.shape[1:] == (64, 64)


def test_phi_series_carries_identifiers_and_private_tag(corpus: Path) -> None:
    entry = next(e for e in _manifest(corpus)["series"] if "phi_series" in e["directory"])
    ds = pydicom.dcmread(next(iter(sorted((corpus / entry["directory"]).glob("*.dcm")))))
    assert ds.PatientName == "Doe^Jane"
    assert ds.PatientID == "MRN-884211"
    assert ds.PatientBirthDate == "19730102"
    assert ds.InstitutionName == "Springfield General Hospital"
    assert ds.ReferringPhysicianName == "House^Gregory"
    assert ds.AccessionNumber == "ACC-99173"
    assert ds.StudyDate == "20231115"
    assert ds[0x00091001].value == "PRIVATE_MRN_884211"
    assert ds[0x00091001].tag.is_private


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = build_fixture_set(tmp_path / "first")
    second = build_fixture_set(tmp_path / "second")
    assert [a.series_uid for a in first] == [a.series_uid for a in second]
    for a, _b in zip(first, second, strict=True):
        rel = a.directory.relative_to(tmp_path / "first")
        ha = hashlib.sha256((a.directory / "slice_000.dcm").read_bytes()).hexdigest()
        hb = hashlib.sha256((tmp_path / "second" / rel / "slice_000.dcm").read_bytes()).hexdigest()
        assert ha == hb
