"""Tests for the QC rule registry and report generation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from dicomvolumeqc.config import load_config
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series, load_volume
from dicomvolumeqc.models import SeriesRef, Severity
from dicomvolumeqc.qc import QCReport, RuleRegistry, Verdict, run_qc


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("qc_corpus")
    build_fixture_set(root)
    return root


@pytest.fixture(scope="module")
def refs(corpus: Path) -> list[SeriesRef]:
    return discover_series(corpus / "raw")


@pytest.fixture(scope="module")
def report(refs: list[SeriesRef]) -> QCReport:
    return run_qc(refs, load_config())


def _ref(refs: list[SeriesRef], name: str) -> SeriesRef:
    return next(ref for ref in refs if ref.directory.name == name)


def _result(report: QCReport, name: str):
    return next(result for result in report.series if result.directory.endswith(name))


def test_registry_exposes_all_rules() -> None:
    registry = RuleRegistry(load_config())
    codes = set(registry.rules)
    assert len(codes) >= 8
    assert {
        "TOO_FEW_SLICES",
        "NONUNIFORM_SPACING",
        "SLICE_GAP",
        "DUPLICATE_SLICES",
        "NON_AXIAL",
        "CORRUPT_FILES",
        "EMPTY_OR_CONSTANT_VOLUME",
        "HU_OUT_OF_RANGE",
        "MASK_GRID_MISMATCH",
        "MASK_EMPTY",
    } <= codes
    assert all(rule.severity in {Severity.FAIL, Severity.WARN} for rule in registry.rules.values())
    assert len(registry.catalog()) == len(codes)


def test_clean_and_phi_series_pass(report: QCReport) -> None:
    for name in (
        "clean_axial_070",
        "clean_axial_080",
        "clean_axial_060",
        "clean_axial_100",
        "clean_axial_125",
        "clean_axial_090",
        "clean_axial_075",
        "clean_axial_065",
        "phi_series",
    ):
        result = _result(report, name)
        assert result.verdict == Verdict.PASS, (name, result.codes)
        assert result.findings == []


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("nonuniform_spacing", "NONUNIFORM_SPACING"),
        ("missing_slice", "SLICE_GAP"),
        ("duplicate_slices", "DUPLICATE_SLICES"),
        ("tilted_orientation", "NON_AXIAL"),
        ("truncated_series", "CORRUPT_FILES"),
        ("single_slice", "TOO_FEW_SLICES"),
        ("extreme_hu", "HU_OUT_OF_RANGE"),
        ("mask_mismatch", "MASK_GRID_MISMATCH"),
    ],
)
def test_injected_defect_maps_to_rule_and_quarantine(
    report: QCReport, name: str, code: str
) -> None:
    result = _result(report, name)
    assert result.verdict == Verdict.QUARANTINE
    assert code in result.fail_codes


def test_aggregate_counts(report: QCReport) -> None:
    assert report.counts == {"total": 17, "pass": 9, "warn": 0, "quarantine": 8}
    failures = report.failures_by_rule
    assert failures["TOO_FEW_SLICES"] == 1
    assert failures["DUPLICATE_SLICES"] == 1
    assert failures["NON_AXIAL"] == 1
    assert failures["CORRUPT_FILES"] == 1
    assert failures["HU_OUT_OF_RANGE"] == 1
    assert failures["MASK_GRID_MISMATCH"] == 2
    assert failures["NONUNIFORM_SPACING"] == 2
    assert failures["SLICE_GAP"] == 2


def test_qc_report_writes_csv_and_json(report: QCReport, tmp_path: Path) -> None:
    csv_path, json_path = report.write(tmp_path)
    frame = pd.read_csv(csv_path)
    assert len(frame) == 17
    assert set(frame["verdict"]) == {"PASS", "QUARANTINE"}
    payload = json.loads(json_path.read_text())
    assert payload["counts"] == report.counts
    assert len(payload["rules"]) == len(report.rule_catalog)
    assert payload["rules"][0]["severity"] in {"FAIL", "WARN"}


def test_empty_and_constant_volume_rule(refs: list[SeriesRef]) -> None:
    registry = RuleRegistry(load_config())
    volume = load_volume(_ref(refs, "clean_axial_080"))
    meta = dataclasses.replace(volume.meta, constant_volume=True)
    findings = registry.run(meta, volume)
    assert [f.code for f in findings] == ["EMPTY_OR_CONSTANT_VOLUME"]
    assert findings[0].severity == Severity.FAIL


def test_mask_empty_and_missing_rules(refs: list[SeriesRef]) -> None:
    registry = RuleRegistry(load_config())
    volume = load_volume(_ref(refs, "clean_axial_080"))
    empty = dataclasses.replace(
        volume.meta, mask_path=volume.meta.mask_path, mask_label_count=0, mask_grid_matches=True
    )
    assert [f.code for f in registry.run(empty, volume)] == ["MASK_EMPTY"]
    missing = dataclasses.replace(volume.meta, mask_path=None, mask_label_count=None)
    findings = registry.run(missing, volume)
    assert [f.code for f in findings] == ["MASK_MISSING"]
    assert findings[0].severity == Severity.WARN


def test_hu_rule_uses_volume_values(refs: list[SeriesRef]) -> None:
    registry = RuleRegistry(load_config())
    extreme = load_volume(_ref(refs, "extreme_hu"))
    findings = registry.run(extreme.meta, extreme)
    assert [f.code for f in findings] == ["HU_OUT_OF_RANGE"]


def test_load_failure_is_quarantined(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "junk.dcm").write_bytes(b"not a dicom file")
    refs = discover_series(tmp_path)
    report = run_qc(refs, load_config())
    assert report.counts["quarantine"] == 1
    result = report.series[0]
    assert result.verdict == Verdict.QUARANTINE
    assert result.codes == ["LOAD_FAILED"]
    assert result.meta is None
    assert "n_slices" not in result.to_dict()


def test_series_result_serialization(report: QCReport) -> None:
    payload = _result(report, "clean_axial_080").to_dict()
    assert payload["n_slices"] == 12
    assert payload["volume_shape"] == [12, 64, 64]
    assert payload["codes"] == []
    assert payload["error"] is None
    json.dumps(payload)
