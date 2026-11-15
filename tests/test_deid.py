"""Tests for PHI de-identification and the independent leak auditor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pydicom
import pytest

from dicomvolumeqc.config import load_config
from dicomvolumeqc.deid import (
    PHI_TAGS,
    collect_patient_ids,
    deidentify_dataset,
    patient_pseudonym,
    scan_corpus,
    scan_for_phi,
)
from dicomvolumeqc.errors import DeidError
from dicomvolumeqc.fixtures import build_fixture_set


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("deid_corpus")
    build_fixture_set(root)
    return root


@pytest.fixture(scope="module")
def deid_run(corpus: Path, tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("deid_output")
    config = load_config()
    original_ids = collect_patient_ids(corpus / "raw")
    summary = deidentify_dataset(
        corpus / "raw", root / "deid", config.deid_salt, root / "crosswalk.csv"
    )
    return root / "deid", root / "crosswalk.csv", summary, original_ids


def _phi_file(corpus: Path) -> Path:
    return sorted((corpus / "raw" / "PAT008" / "phi_series").glob("*.dcm"))[0]


def test_deid_summary_counts(deid_run, corpus: Path) -> None:
    _deid, crosswalk, summary, original_ids = deid_run
    assert summary.files_processed == 206
    assert summary.series_count == 17
    assert summary.patient_count == 8
    assert summary.masks_copied == 17
    assert summary.tags_remapped == 618
    assert summary.private_tags_removed == 24
    assert summary.tags_blanked > 0
    assert len(original_ids) == 8
    assert crosswalk.is_file()
    assert not crosswalk.is_relative_to(Path(summary.destination))


def test_scan_finds_no_leaks_after_deidentification(deid_run) -> None:
    deid, _crosswalk, _summary, original_ids = deid_run
    assert scan_corpus(deid, original_ids=original_ids) == []


def test_raw_phi_fixture_is_a_positive_control(deid_run, corpus: Path) -> None:
    leaks = scan_for_phi(_phi_file(corpus))
    assert len(leaks) > 0
    reasons = {leak.reason for leak in leaks}
    assert "PHI_TAG" in reasons
    assert "PRIVATE_TAG" in reasons
    assert any(leak.tag == "PatientName" for leak in leaks)
    raw_leaks = scan_corpus(corpus / "raw")
    assert len(raw_leaks) > 100


def test_residual_original_id_detection(deid_run, corpus: Path) -> None:
    deid, _crosswalk, _summary, original_ids = deid_run
    leaks = scan_corpus(deid, original_ids=original_ids)
    assert not any(leak.reason == "RESIDUAL_ORIGINAL_ID" for leak in leaks)
    one = sorted((deid / "PAT008" / "phi_series").glob("*.dcm"))[0]
    raw = one.read_bytes()
    assert b"MRN-884211" not in raw


def test_pseudonyms_stable_across_runs(
    corpus: Path, deid_run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    deid, _crosswalk, _summary, _ids = deid_run
    second_root = tmp_path_factory.mktemp("deid_second")
    deidentify_dataset(
        corpus / "raw",
        second_root / "deid",
        load_config().deid_salt,
        second_root / "crosswalk.csv",
    )
    first = pydicom.dcmread(sorted((deid / "PAT001").rglob("*.dcm"))[0], stop_before_pixels=True)
    second = pydicom.dcmread(
        sorted((second_root / "deid" / "PAT001").rglob("*.dcm"))[0], stop_before_pixels=True
    )
    assert first.PatientID == second.PatientID
    assert first.StudyInstanceUID == second.StudyInstanceUID
    assert first.SeriesInstanceUID == second.SeriesInstanceUID
    assert first.SOPInstanceUID == second.SOPInstanceUID


def test_pseudonyms_differ_across_patients_and_salts(deid_run) -> None:
    deid, _crosswalk, _summary, _ids = deid_run
    one = pydicom.dcmread(sorted((deid / "PAT001").rglob("*.dcm"))[0], stop_before_pixels=True)
    two = pydicom.dcmread(sorted((deid / "PAT002").rglob("*.dcm"))[0], stop_before_pixels=True)
    assert one.PatientID != two.PatientID
    assert patient_pseudonym("MRN-PAT001", "salt-a") != patient_pseudonym("MRN-PAT001", "salt-b")


def test_pixel_data_is_byte_identical(deid_run, corpus: Path) -> None:
    deid, _crosswalk, _summary, _ids = deid_run
    relative_paths = [
        "PAT008/phi_series/slice_000.dcm",
        "PAT001/clean_axial_070/slice_000.dcm",
        "PAT001/truncated_series/slice_000.dcm",
        "PAT003/extreme_hu/slice_005.dcm",
    ]
    for relative in relative_paths:
        before = pydicom.dcmread(corpus / "raw" / relative).PixelData
        after = pydicom.dcmread(deid / relative).PixelData
        assert hashlib.sha256(before).hexdigest() == hashlib.sha256(after).hexdigest()


def test_deidentified_header_is_clean(deid_run) -> None:
    deid, _crosswalk, _summary, _ids = deid_run
    ds = pydicom.dcmread(sorted((deid / "PAT008" / "phi_series").glob("*.dcm"))[0])
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.DeidentificationMethod.startswith("dvqc-hmac-sha256-v1")
    assert len(ds.DeidentificationMethod) <= 64
    assert ds.PatientName == ""
    assert ds.PatientBirthDate == ""
    assert ds.StudyDate == ""
    assert ds.AccessionNumber == ""
    assert ds.InstitutionName == ""
    assert ds.ReferringPhysicianName == ""
    assert str(ds.PatientID).startswith("PSN-")
    assert str(ds.StudyInstanceUID).startswith("2.25.")
    assert not any(element.tag.is_private for element in ds.iterall())
    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
    assert "PatientName" in PHI_TAGS


def test_crosswalk_contains_originals_and_pseudonyms(deid_run) -> None:
    deid, crosswalk, _summary, _ids = deid_run
    text = crosswalk.read_text()
    assert "MRN-884211" in text
    assert "PSN-" in text
    assert "entity,original,pseudonym" in text
    assert not list(deid.rglob("*.csv"))


def test_full_precision_date_is_flagged(deid_run, tmp_path: Path) -> None:
    deid, _crosswalk, _summary, _ids = deid_run
    ds = pydicom.dcmread(sorted((deid / "PAT008" / "phi_series").glob("*.dcm"))[0])
    ds.add_new((0x0018, 0x1200), "DA", "20240101")
    target = tmp_path / "date_leak.dcm"
    ds.save_as(target, enforce_file_format=True)
    reasons = {leak.reason for leak in scan_for_phi(target)}
    assert "FULL_PRECISION_DATE" in reasons


def test_crosswalk_inside_dataset_is_rejected(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(DeidError, match="inside the de-identified dataset"):
        deidentify_dataset(
            corpus / "raw",
            tmp_path / "deid",
            "salt",
            tmp_path / "deid" / "crosswalk.csv",
        )


def test_deid_rejects_empty_salt_and_missing_source(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(DeidError, match="salt"):
        deidentify_dataset(corpus / "raw", tmp_path / "deid", "", tmp_path / "cw.csv")
    with pytest.raises(DeidError, match="does not exist"):
        deidentify_dataset(tmp_path / "missing", tmp_path / "deid", "salt", tmp_path / "cw.csv")


def test_scan_unreadable_file_returns_leak(tmp_path: Path) -> None:
    path = tmp_path / "junk.dcm"
    path.write_bytes(b"junk")
    leaks = scan_for_phi(path)
    assert len(leaks) == 1
    assert leaks[0].reason == "UNREADABLE_FILE"
