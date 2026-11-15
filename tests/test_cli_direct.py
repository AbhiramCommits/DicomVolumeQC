"""In-process CLI tests that also contribute to coverage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dicomvolumeqc.cli import main


@pytest.fixture(scope="module")
def run_all(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("cli_direct")
    code = main(
        [
            "run-all",
            "--out",
            str(root / "fixtures"),
            "--reports",
            str(root / "reports"),
            "--export",
            str(root / "export"),
        ]
    )
    return root, code


def test_run_all_direct(run_all) -> None:
    root, code = run_all
    assert code == 0
    assert (root / "export" / "manifest.json").is_file()
    assert (root / "reports" / "run_all.txt").is_file()
    assert (root / "reports" / "deid_report.json").is_file()
    assert (root / "reports" / "pipeline_summary.json").is_file()


def test_verify_direct(run_all) -> None:
    root, _code = run_all
    assert main(["verify", "--export", str(root / "export")]) == 0


def test_verify_json_direct(run_all, capsys: pytest.CaptureFixture[str]) -> None:
    root, _code = run_all
    assert main(["verify", "--export", str(root / "export"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_verify_tampered_direct(run_all, tmp_path: Path) -> None:
    root, _code = run_all
    tampered = tmp_path / "export"
    shutil.copytree(root / "export", tampered)
    target = next((tampered / "train").rglob("*.npy"))
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(data)
    assert main(["verify", "--export", str(tampered)]) == 1


def test_ingest_qc_prepare_direct(run_all) -> None:
    root, _code = run_all
    assert main(["ingest", "--input", str(root / "fixtures" / "raw")]) == 0
    assert (
        main(
            [
                "qc",
                "--input",
                str(root / "fixtures" / "deid"),
                "--reports",
                str(root / "qc2"),
            ]
        )
        == 0
    )
    assert main(["prepare", "--input", str(root / "fixtures" / "deid")]) == 0


def test_fixtures_and_deid_direct(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    assert main(["fixtures", "--out", str(fixtures)]) == 0
    deid = tmp_path / "deid"
    crosswalk = tmp_path / "reports" / "crosswalk.csv"
    assert (
        main(
            [
                "deid",
                "--input",
                str(fixtures / "raw"),
                "--out",
                str(deid),
                "--crosswalk",
                str(crosswalk),
                "--reports",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    assert crosswalk.is_file()
    payload = json.loads((tmp_path / "reports" / "deid_report.json").read_text())
    assert payload["leaks_after"] == 0


def test_export_direct(run_all, tmp_path: Path) -> None:
    root, _code = run_all
    assert (
        main(
            [
                "export",
                "--input",
                str(root / "fixtures" / "deid"),
                "--out",
                str(tmp_path / "export"),
                "--reports",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )
    assert (tmp_path / "export" / "manifest.json").is_file()


def test_bad_config_returns_two(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("target_spacing_mm: 0\n")
    assert main(["fixtures", "--out", str(tmp_path / "f"), "--config", str(bad)]) == 2


def test_empty_ingest_returns_one(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["ingest", "--input", str(empty)]) == 1


def test_seed_override(run_all, tmp_path: Path) -> None:
    root, _code = run_all
    assert (
        main(
            [
                "export",
                "--input",
                str(root / "fixtures" / "deid"),
                "--out",
                str(tmp_path / "export"),
                "--reports",
                str(tmp_path / "reports"),
                "--seed",
                "99",
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "export" / "manifest.json").read_text())
    assert manifest["split_seed"] == 99
