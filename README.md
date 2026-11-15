# DicomVolumeQC

DicomVolumeQC is a reproducible, offline pipeline that turns DICOM CT series into a
de-identified, quality-controlled, ML-ready 3D dataset. It reconstructs volumes from
slice geometry, quarantines series that violate spatial or intensity invariants,
strips PHI with salted pseudonyms and an independent leak audit, resamples to
isotropic spacing with CT windowing, aligns segmentation masks, makes patient-disjoint
train/val/test splits, and exports every volume and mask with a SHA-256 manifest that
can be re-verified later. That chain of custody — config hash, commit, seed, per-file
checksums, and a top-level manifest digest — is the part that matters when the data
preparation path is an input to regulated software (SaMD) rather than an ad-hoc script.

## Architecture

| Module | Responsibility |
|---|---|
| `fixtures.py`, `fixture_study.py` | Deterministic synthetic CT corpus with 17 series across 8 patients; defects are injected by named flags. |
| `ingest.py`, `geometry.py` | Discover series by `SeriesInstanceUID`, reconstruct HU volumes sorted by slice-normal projection, and emit structured geometry findings. |
| `qc.py`, `qc_rules.py` | Rule registry, per-series PASS/WARN/QUARANTINE verdicts, aggregate counts, CSV/JSON reports. |
| `deid.py`, `phi_audit.py` | HMAC-derived pseudonyms, PHI blanking, private-tag removal, crosswalk, and an independent leak scanner. |
| `preprocess.py` | CT windowing, isotropic SimpleITK resampling, mask resampling/alignment, round-trip Dice. |
| `splits.py` | Whole-patient, seeded train/val/test assignment with achieved counts. |
| `manifest.py` | Export `.npy` volumes/masks, per-file SHA-256, manifest digest, and checksum verification. |
| `pipeline.py`, `cli.py`, `cli_commands.py` | End-to-end orchestration and the `dvqc` command line. |

```text
DICOM series + binary masks
        │  deid: salted pseudonyms, PHI/date/private-tag removal, leak audit
        ▼
de-identified DICOM + masks
        │  ingest: group by SeriesInstanceUID, sort by slice normal, rescale to HU
        ▼
Volume + SeriesMeta ──► QC registry ──► reports/qc_report.{csv,json}
        │ PASS / WARN only
        │  resample to 1.0 mm isotropic ──► window ──► align mask (nearest)
        ▼
PreparedVolume
        │  patient-disjoint split (seeded)
        ▼
export/{split}/{patient}/{series}/volume.npy + mask.npy
        │  SHA-256 per file + canonical manifest digest
        ▼
export/manifest.{json,csv} ──► dvqc verify
```

### QC rule table

All thresholds live in `configs/default.yaml` and are shared with the geometry
validator. Expanded rationale is in [`docs/QC_RULES.md`](docs/QC_RULES.md).

| Code | Severity | What it detects |
|---|---|---|
| `TOO_FEW_SLICES` | FAIL | Fewer than 3 slices; no usable through-plane extent. |
| `NONUNIFORM_SPACING` | FAIL | Slice-spacing coefficient of variation above 0.05. |
| `SLICE_GAP` | FAIL | A gap larger than 1.5× the median spacing (dropped slice). |
| `DUPLICATE_SLICES` | FAIL | Repeated pixel payloads found by SHA-256 of each slice. |
| `NON_AXIAL` | FAIL | Slice normal more than 5° from the axial axis. |
| `ORIENTATION_NOT_ORTHONORMAL` | FAIL | Direction cosines not unit length and mutually orthogonal. |
| `CORRUPT_FILES` | FAIL | Header readable but PixelData short/undecodable. |
| `EMPTY_OR_CONSTANT_VOLUME` | FAIL | Zero voxels or zero intensity range. |
| `HU_OUT_OF_RANGE` | FAIL | Voxels outside [-1024, 3071] HU. |
| `INCONSISTENT_GEOMETRY` | FAIL | Rows/Columns, PixelSpacing, or orientation varies across slices. |
| `MASK_GRID_MISMATCH` | FAIL | Mask shape/spacing/origin/direction differs from the image grid. |
| `MASK_EMPTY` | FAIL | Mask has zero labeled voxels. |
| `MASK_MISSING` | WARN | No mask is available; image-only export remains possible. |
| `LOAD_FAILED` | FAIL | Volume reconstruction raised `GeometryError`. |
| `MASK_UNREADABLE` | FAIL | Mask file cannot be read as a 3D volume. |
| `MASK_NOT_BINARY` | WARN | Mask labels outside `{0, 1}`. |

A series with any FAIL is `QUARANTINE` and is never exported. A WARN-only series is
exported with its findings recorded in the manifest. PASS/WARN/QUARANTINE and the
per-rule counts are written to `reports/qc_report.csv` and `reports/qc_report.json`.

## How to run

Requires Python 3.11 or 3.12; everything runs offline.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install                 # pip install -e ".[dev]"
make fixtures                # write data/fixtures/raw + masks
dvqc run-all                 # fixtures -> deid -> ingest -> qc -> prepare -> export -> verify
dvqc verify --export export  # recompute every checksum
make test                    # pytest with the 85% coverage gate
```

Defaults are `--out data/fixtures --reports reports --export export`; every command
also accepts `--config`, `--seed`, and `--json`. The full run writes
`reports/run_all.txt`, `reports/qc_report.{csv,json}`,
`reports/deid_report.json`, `reports/pipeline_summary.json`, and the checksummed
`export/manifest.json`.

To point the pipeline at a real DICOM tree instead of the fixtures:

```bash
dvqc deid --input /path/to/dicom_root --out data/real_deid \
  --crosswalk reports/real_deid_crosswalk.csv
dvqc qc   --input data/real_deid --reports reports/real
dvqc export --input data/real_deid --out export_real --reports reports/real
dvqc verify --export export_real
```

Masks are expected as `<series>_mask.nii.gz` beside each series; a missing mask is a
WARN, while a mismatched grid or empty mask is a FAIL. The crosswalk contains original
identifiers and is deliberately written outside the de-identified tree (and is
gitignored). Review any real-data run against your institution's policy; this project
is not a certified medical device.

## Results

Every figure below comes from this build's `reports/metrics.json`,
`reports/pipeline_summary.json`, `reports/run_all.txt`, and the generated manifest
(measured code commit `2896454`), not from estimates.

### Corpus and QC

| Metric | Measured value |
|---|---|
| Synthetic series / patients | 17 / 8 |
| Series ingested | 17 |
| QC PASS | 9 |
| QC WARN | 0 |
| QC QUARANTINE | 8 |
| Series passing QC and exported | 9 |
| Mask-image round-trip Dice (9 exported masks) | min 0.939743, median 0.959369, max 0.979642 |

Quarantined series by rule code (one series can trigger several rules):

| Rule code | Quarantined series |
|---|---:|
| `CORRUPT_FILES` | 1 |
| `DUPLICATE_SLICES` | 1 |
| `HU_OUT_OF_RANGE` | 1 |
| `MASK_GRID_MISMATCH` | 2 |
| `NONUNIFORM_SPACING` | 2 |
| `NON_AXIAL` | 1 |
| `SLICE_GAP` | 2 |
| `TOO_FEW_SLICES` | 1 |

The 12 rule hits map to 8 quarantined series: the missing-slice fixture also has
non-uniform spacing, and the truncated fixture also mismatches its mask grid.

### De-identification

| Metric | Measured value |
|---|---|
| DICOM files processed | 206 |
| Series / patients pseudonymized | 17 / 8 |
| PHI fields blanked | 1,866 |
| UIDs remapped (Study/Series/SOP) | 618 |
| Private tags removed | 24 |
| PHI leaks before de-identification | 2,714 (2,690 `PHI_TAG`, 24 `PRIVATE_TAG`) |
| PHI leaks after de-identification | 0 |

Pseudonyms are HMAC-SHA-256(salt, kind, original) values: the same patient maps to the
same `PSN-…` value across series and runs, while different patients differ. The leak
auditor re-reads the output independently and checks surviving PHI tags, private tags,
full-precision dates, and residual original PatientID bytes. A round-trip test also
proves exported PixelData is byte-identical to the source.

### Isotropic preparation

Target spacing is **1.0 mm** in all axes with the `soft_tissue` window (center 40 HU,
width 400 HU) and linear image interpolation. Example transformations from the exported
records:

| Source shape (z, y, x) | Source spacing (x, y, z) mm | Prepared shape | Prepared spacing | Mask Dice |
|---|---|---|---|---:|---:|
| 12 × 64 × 64 | 0.9 × 0.9 × 2.5 | 30 × 58 × 58 | 1.0 × 1.0 × 1.0 | 0.953760 |
| 20 × 64 × 64 | 0.6 × 0.6 × 2.5 | 50 × 38 × 38 | 1.0 × 1.0 × 1.0 | 0.969739 |
| 18 × 64 × 64 | 0.65 × 0.65 × 2.5 | 45 × 42 × 42 | 1.0 × 1.0 × 1.0 | 0.958524 |

Windowed output is float32 and bounded in `[0, 1]`. Masks remain strictly binary
through nearest-neighbour resampling; the test gate requires round-trip Dice > 0.95 on
the 0.8 mm fixture, which measured 0.959369. The lowest corpus value, 0.939743, belongs
to a 14-slice 0.75 mm fixture and is reported rather than hidden.

### Split and export

| Split | Patients | Series |
|---|---:|---:|
| train | 4 | 5 |
| val | 2 | 2 |
| test | 2 | 2 |

No patient appears in more than one split. Prior to splitting, 8 quarantined series are
excluded, leaving 9 exported series (8 patients).

| Metric | Measured value |
|---|---|
| Exported records | 9 |
| Exported files (volume + mask) | 18 |
| Manifest verification | `ok=True`, `digest_ok=True` |
| Config hash | `fa05c3b2f7e74dd5d93f40c03c8ff44da5432cc619b5a3eb4126a1f808ff34d0` |
| End-to-end wall clock | 3.126 s |
| Pipeline-internal runtime | 2.201878 s |

`dvqc verify` re-hashes all 18 files and recomputes the canonical 64-character
manifest digest. The digest signs the manifest's own `generated_at` and `git_commit`
fields, so it intentionally changes between runs and commits even when the data bytes
are identical; per-file checksums remain stable. The current values, including the
frozen build's digest, are in `reports/metrics.json`.

### Test suite

| Metric | Measured value |
|---|---|
| Tests passing | 145 |
| Line coverage | 96.3724% (1,594 / 1,654 statements) |
| Coverage gate | 85% |

Coverage and tests were measured with
`pytest --cov=dicomvolumeqc --cov-report=json:reports/coverage.json --cov-fail-under=85`.
The suite covers every injected defect, patient disjointness, deterministic splits,
manifest tampering, PHI leak negative controls, unreadable inputs, bad configs, and CLI
error paths.

## Design notes and limitations

**Why synthetic fixtures.** A regulated data-preparation repository must be runnable
offline, deterministically, and without touching real PHI. The corpus is small
(64 × 64 in-plane, 17 series, about 2.5 MB) so the complete pipeline finishes in
seconds on a laptop. Every defect is injected by a named flag, giving the QC tests
ground truth instead of hand-labeled guesses. All identifiers are fabricated.

**What a real clinical dataset adds.** Real archives bring multi-frame and enhanced
DICOM objects, localizers, contrast phases, metal artifacts, varying reconstruction
kernels and slice thicknesses, non-axial acquisitions, mixed transfer syntaxes,
burned-in pixel PHI, private vendor tags, RTSTRUCT/DICOM-SEG masks, missing or
misaligned segmentations, site/scanner effects, and much larger volumes. The
de-identifier here handles tag-level PHI, private tags, dates, and residual PatientID
bytes; it does **not** detect burned-in text in pixels or perform defacing. A real
deployment needs DICOM PS3.15 profile validation, OCR/visual review for burned-in PHI,
and an approved data-use/IRB path.

**What a regulated pipeline would add.** Traceability here is deliberately concrete:
config hash, git commit, seed, per-file SHA-256, manifest digest, and reproducible
reports. A production SaMD data path would additionally need a quality management
system, documented intended use, ISO 14971 risk analysis, IEC 62304 software lifecycle
records, formal verification/validation, access control and audit logs, change control
with review, retention/disaster-recovery plans, and expert clinical review of QC
thresholds. This repository demonstrates the mechanics and the review discipline, not
a certified device.

**Known limitations.** No PACS/DICOMweb integration, database, model training, or UI.
De-identification is not a substitute for institutional review. Date removal is used
instead of date shifting, which can remove clinically useful temporal context.
Patient-level splitting can produce unequal series counts when one patient has many
series. The synthetic corpus is not evidence of clinical performance.

## Repository layout

```text
configs/default.yaml        QC tolerances, spacing, window, split ratios, seed
src/dicomvolumeqc/          library and CLI
tests/                      unit, integration, CLI, and error-path tests
scripts/record_metrics.py   rebuilds reports/metrics.json from measured artifacts
docs/QC_RULES.md            expanded rule rationale
reports/                    measured run outputs (crosswalk is gitignored)
export/                     generated volumes, masks, manifest (gitignored)
.github/workflows/ci.yml    lint/test matrix + end-to-end artifact job
```

## License

MIT. See [`LICENSE`](LICENSE). This software is for research and engineering work on
synthetic or properly governed data; it is not a medical device.
