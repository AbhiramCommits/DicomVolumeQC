"""Subprocess smoke tests for the dvqc CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dicomvolumeqc.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("cli_e2e")
    fixture_root = root / "fixtures"
    reports = root / "reports"
    export = root / "export"
    result = _run(
        [
            "run-all",
            "--out",
            str(fixture_root),
            "--reports",
            str(reports),
            "--export",
            str(export),
        ]
    )
    return root, fixture_root, reports, export, result


def test_run_all_succeeds_and_writes_artifacts(pipeline_run) -> None:
    _root, _fixtures, reports, export, result = pipeline_run
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verify: ok=True" in result.stdout
    assert (export / "manifest.json").is_file()
    assert (export / "manifest.csv").is_file()
    assert (reports / "run_all.txt").is_file()
    assert (reports / "qc_report.csv").is_file()
    assert (reports / "pipeline_summary.json").is_file()
    manifest = json.loads((export / "manifest.json").read_text())
    assert manifest["counts"]["series"] == 9
    log = (reports / "run_all.txt").read_text()
    assert "fixtures: 17 series, 8 patients" in log


def test_verify_subcommand_passes_and_supports_json(pipeline_run) -> None:
    _root, _fixtures, _reports, export, _result = pipeline_run
    result = _run(["verify", "--export", str(export), "--json"])
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["files_checked"] == 18


def test_verify_subcommand_detects_tampering(pipeline_run, tmp_path: Path) -> None:
    _root, _fixtures, _reports, export, _result = pipeline_run
    tampered = tmp_path / "export"
    shutil.copytree(export, tampered)
    manifest = json.loads((tampered / "manifest.json").read_text())
    target = tampered / manifest["records"][0]["volume_path"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(data)
    result = _run(["verify", "--export", str(tampered)])
    assert result.returncode != 0
    assert "checksum mismatch" in result.stdout


def test_empty_input_returns_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run(["ingest", "--input", str(empty)])
    assert result.returncode != 0
    assert "0/0 series" in result.stdout
