"""Tests for patient-disjoint splitting."""

from __future__ import annotations

import dataclasses

import pytest

from dicomvolumeqc.config import load_config
from dicomvolumeqc.errors import ConfigError, SplitError
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series
from dicomvolumeqc.qc import Verdict, run_qc
from dicomvolumeqc.splits import SPLITS, patient_disjoint_split


@pytest.fixture(scope="module")
def passing_metas(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("split_corpus")
    build_fixture_set(root)
    refs = discover_series(root / "raw")
    report = run_qc(refs, load_config())
    metas = [r.meta for r in report.series if r.verdict != Verdict.QUARANTINE and r.meta]
    assert len(metas) == 9
    return metas


def test_patient_disjoint_assignments(passing_metas) -> None:
    result = patient_disjoint_split(passing_metas, load_config().split_ratio_map, 7)
    assert set(result.assignments) == {meta.series_uid for meta in passing_metas}
    assert sum(result.series_per_split.values()) == 9
    assert sum(result.patients_per_split.values()) == 8
    assert all(result.patients_per_split[split] > 0 for split in SPLITS)
    for meta in passing_metas:
        assert result.patient_assignment[meta.patient_id] == result.assignments[meta.series_uid]
    assert all(split in SPLITS for split in result.assignments.values())


def test_split_is_deterministic_for_a_fixed_seed(passing_metas) -> None:
    first = patient_disjoint_split(passing_metas, load_config().split_ratio_map, 123)
    second = patient_disjoint_split(passing_metas, load_config().split_ratio_map, 123)
    assert first.assignments == second.assignments
    assert first.patient_assignment == second.patient_assignment
    assert first.counts() == second.counts()


def test_split_accepts_config_object(passing_metas) -> None:
    config = load_config()
    assert (
        patient_disjoint_split(passing_metas, config, config.seed).counts()
        == patient_disjoint_split(passing_metas, config.split_ratio_map, config.seed).counts()
    )


def test_split_summary_has_counts_and_patients(passing_metas) -> None:
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    summary = patient_disjoint_split(passing_metas, ratios, 1).summary()
    assert set(summary["counts"]) == set(SPLITS)
    assert len(summary["patients"]) == 8


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.6, "val": 0.2, "test": 0.1},
        {"train": 0.7, "val": 0.2},
        {"train": 0.8, "val": 0.3, "test": -0.1},
    ],
)
def test_invalid_ratios_rejected(passing_metas, ratios) -> None:
    with pytest.raises(ConfigError):
        patient_disjoint_split(passing_metas, ratios, 1)


def test_empty_input_rejected() -> None:
    with pytest.raises(SplitError, match="empty"):
        patient_disjoint_split([], load_config().split_ratio_map, 1)


def test_missing_patient_id_rejected(passing_metas) -> None:
    metas = [dataclasses.replace(passing_metas[0], patient_id="")]
    with pytest.raises(SplitError, match="patient identifier"):
        patient_disjoint_split(metas, load_config().split_ratio_map, 1)


def test_duplicate_series_rejected(passing_metas) -> None:
    with pytest.raises(SplitError, match="duplicate series"):
        patient_disjoint_split(passing_metas + [passing_metas[0]], load_config().split_ratio_map, 1)


def test_single_patient_split_is_still_disjoint(passing_metas) -> None:
    metas = [meta for meta in passing_metas if meta.patient_id == passing_metas[0].patient_id]
    result = patient_disjoint_split(metas, load_config().split_ratio_map, 1)
    populated = [split for split, count in result.patients_per_split.items() if count]
    assert populated == ["train"]
    assert result.series_per_split["train"] == len(metas)
