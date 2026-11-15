"""Tests for export and checksummed manifest verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dicomvolumeqc.config import load_config
from dicomvolumeqc.errors import DvqcError, QCFailure
from dicomvolumeqc.fixtures import build_fixture_set
from dicomvolumeqc.ingest import discover_series
from dicomvolumeqc.manifest import export_dataset, verify_manifest
from dicomvolumeqc.pipeline import prepare_passing
from dicomvolumeqc.preprocess import preprocess_series
from dicomvolumeqc.qc import Verdict, run_qc
from dicomvolumeqc.splits import patient_disjoint_split


@pytest.fixture(scope="module")
def export_setup(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("manifest_corpus")
    build_fixture_set(root)
    config = load_config()
    refs = discover_series(root / "raw")
    report = run_qc(refs, config)
    prepared = prepare_passing(refs, report, config)
    split = patient_disjoint_split(
        [item.meta for item in prepared], config.split_ratio_map, config.seed
    )
    destination = root / "export"
    result = export_dataset(prepared, split, report, config, destination)
    return {
        "root": root,
        "config": config,
        "refs": refs,
        "report": report,
        "prepared": prepared,
        "split": split,
        "destination": destination,
        "result": result,
    }


def test_manifest_is_written_with_expected_fields(export_setup) -> None:
    destination = export_setup["destination"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0"
    assert isinstance(manifest["generated_at"], str)
    assert len(manifest["git_commit"]) >= 7
    assert len(manifest["config_hash"]) == 64
    assert len(manifest["manifest_digest"]) == 64
    assert manifest["counts"]["series"] == 9
    assert manifest["counts"]["files"] == 18
    assert len(manifest["records"]) == 9
    for record in manifest["records"]:
        assert record["split"] in {"train", "val", "test"}
        assert record["voxel_shape"][0] > 0
        assert len(record["spacing_mm"]) == 3
        assert all(abs(v - 1.0) < 1e-6 for v in record["spacing_mm"])
        assert len(record["sha256"]["volume"]) == 64
        assert len(record["sha256"]["mask"]) == 64
        assert record["qc_verdict"] in {"PASS", "WARN"}
        assert (destination / record["volume_path"]).is_file()
        assert (destination / record["mask_path"]).is_file()


def test_manifest_csv_has_one_row_per_record(export_setup) -> None:
    frame = pd.read_csv(export_setup["destination"] / "manifest.csv")
    assert len(frame) == 9
    assert set(frame["split"]) <= {"train", "val", "test"}


def test_exported_arrays_match_prepared_volumes(export_setup) -> None:
    destination = export_setup["destination"]
    for item in export_setup["prepared"]:
        record = next(
            r
            for r in json.loads((destination / "manifest.json").read_text())["records"]
            if r["series_pseudonym"] == item.series_uid
        )
        volume = np.load(destination / record["volume_path"])
        mask = np.load(destination / record["mask_path"])
        assert volume.dtype == np.float32
        assert float(volume.min()) >= 0.0
        assert float(volume.max()) <= 1.0
        assert set(np.unique(mask)) <= {0, 1}
        assert list(volume.shape) == record["voxel_shape"]


def test_verify_manifest_passes_on_fresh_export(export_setup) -> None:
    verification = verify_manifest(export_setup["destination"])
    assert verification.ok
    assert verification.digest_ok
    assert verification.files_checked == 18
    assert verification.problems == ()


def test_verify_manifest_detects_flipped_byte(export_setup, tmp_path: Path) -> None:
    copied = tmp_path / "export"
    import shutil

    shutil.copytree(export_setup["destination"], copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    target = copied / manifest["records"][0]["volume_path"]
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(data)
    verification = verify_manifest(copied)
    assert not verification.ok
    assert any("checksum mismatch" in problem for problem in verification.problems)


def test_verify_manifest_detects_missing_file(export_setup, tmp_path: Path) -> None:
    import shutil

    copied = tmp_path / "export"
    shutil.copytree(export_setup["destination"], copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    (copied / manifest["records"][0]["mask_path"]).unlink()
    verification = verify_manifest(copied)
    assert not verification.ok
    assert any("missing file" in problem for problem in verification.problems)


def test_verify_manifest_detects_digest_tampering(export_setup, tmp_path: Path) -> None:
    import shutil

    copied = tmp_path / "export"
    shutil.copytree(export_setup["destination"], copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["manifest_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2))
    verification = verify_manifest(copied)
    assert not verification.ok
    assert not verification.digest_ok
    assert "manifest digest mismatch" in verification.problems


def test_verify_manifest_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(DvqcError, match="manifest not found"):
        verify_manifest(tmp_path)


def test_export_refuses_quarantined_series(export_setup, tmp_path: Path) -> None:
    config = export_setup["config"]
    refs = export_setup["refs"]
    report = export_setup["report"]
    prepared = list(export_setup["prepared"])
    quarantined = next(
        ref
        for ref in refs
        if next(r for r in report.series if r.series_uid == ref.series_uid).verdict
        == Verdict.QUARANTINE
    )
    prepared.append(preprocess_series(quarantined, config))
    with pytest.raises(QCFailure, match="quarantined"):
        export_dataset(prepared, export_setup["split"], report, config, tmp_path / "refused")


def test_export_without_mask_writes_one_file(export_setup, tmp_path: Path) -> None:
    import dataclasses

    maskless = dataclasses.replace(export_setup["prepared"][0], mask=None)
    result = export_dataset(
        [maskless],
        export_setup["split"],
        export_setup["report"],
        export_setup["config"],
        tmp_path / "maskless",
    )
    assert result.files == 1
    manifest = json.loads((tmp_path / "maskless" / "manifest.json").read_text())
    assert manifest["records"][0]["mask_path"] is None
    verification = verify_manifest(tmp_path / "maskless")
    assert verification.ok
    assert verification.files_checked == 1


def test_export_rejects_unsafe_patient_component(export_setup, tmp_path: Path) -> None:
    import dataclasses

    prepared = dataclasses.replace(
        export_setup["prepared"][0],
        meta=dataclasses.replace(export_setup["prepared"][0].meta, patient_id="../evil"),
    )
    with pytest.raises(DvqcError, match="unsafe patient"):
        export_dataset(
            [prepared],
            export_setup["split"],
            export_setup["report"],
            export_setup["config"],
            tmp_path / "unsafe",
        )


def test_export_requires_qc_result(export_setup, tmp_path: Path) -> None:
    import dataclasses

    item = export_setup["prepared"][0]
    report = dataclasses.replace(
        export_setup["report"],
        series=[s for s in export_setup["report"].series if s.series_uid != item.series_uid],
    )
    with pytest.raises(DvqcError, match="no QC result"):
        export_dataset(
            [item], export_setup["split"], report, export_setup["config"], tmp_path / "noqc"
        )


def test_export_requires_split_assignment(export_setup, tmp_path: Path) -> None:
    from dicomvolumeqc.splits import SplitResult

    empty_split = SplitResult(
        assignments={},
        patient_assignment={},
        patients_per_split={"train": 0, "val": 0, "test": 0},
        series_per_split={"train": 0, "val": 0, "test": 0},
    )
    with pytest.raises(DvqcError, match="no split assignment"):
        export_dataset(
            [export_setup["prepared"][0]],
            empty_split,
            export_setup["report"],
            export_setup["config"],
            tmp_path / "nosplit",
        )


def test_verify_manifest_handles_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(DvqcError, match="could not read manifest"):
        verify_manifest(tmp_path)
