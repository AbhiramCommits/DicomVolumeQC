# QC rule reference

Every series is evaluated by the registry in `src/dicomvolumeqc/qc_rules.py`
against `configs/default.yaml`. A rule returns structured findings
(`code`, `severity`, `message`, `observed`) rather than raising, so the report
can explain *why* a series was quarantined.

## Verdict policy

| Findings | Verdict | Export |
|---|---|---|
| Any `FAIL` | `QUARANTINE` | Excluded |
| No `FAIL`, at least one `WARN` | `WARN` | Exported, flagged in the manifest |
| No findings | `PASS` | Exported |

A series can carry several findings; the per-rule counts therefore sum to more
than the number of quarantined series.

## Registry rules

| Code | Severity | Detects | Default threshold |
|---|---|---|---|
| `TOO_FEW_SLICES` | FAIL | A series with no usable through-plane extent (including the single-slice fixture). | `qc.min_slices: 3` |
| `NONUNIFORM_SPACING` | FAIL | Jittered or inconsistent slice positions that break the regular-grid assumption. | spacing CV > `qc.spacing_cv_tolerance: 0.05` |
| `SLICE_GAP` | FAIL | A dropped slice, visible as a gap in projected slice positions. | gap > `qc.gap_factor: 1.5` × median spacing |
| `DUPLICATE_SLICES` | FAIL | Repeated pixel payloads (SHA-256 of each slice) that would bias training by duplication. | any duplicate |
| `NON_AXIAL` | FAIL | Tilted acquisition geometry that the axial resampling path cannot safely normalize. | angle > `qc.axial_tolerance_deg: 5.0` |
| `ORIENTATION_NOT_ORTHONORMAL` | FAIL | Direction cosines that are not unit length and mutually orthogonal. | 1e-3 tolerance |
| `CORRUPT_FILES` | FAIL | Files whose headers read but whose PixelData cannot be decoded (short/truncated payload). | any unreadable slice |
| `EMPTY_OR_CONSTANT_VOLUME` | FAIL | A volume with zero voxels or zero intensity range (no learnable signal). | exact |
| `HU_OUT_OF_RANGE` | FAIL | Voxel values outside a plausible CT HU range (metal, scaling overflow, bad intercept). | outside `[-1024, 3071]` HU |
| `INCONSISTENT_GEOMETRY` | FAIL | Rows/Columns, PixelSpacing, or orientation differ across slices in one series. | any variation |
| `MASK_GRID_MISMATCH` | FAIL | Mask shape, spacing, origin, or direction differs from the image grid. | 1e-4 tolerance |
| `MASK_EMPTY` | FAIL | A mask with no labeled voxels (nothing to supervise). | foreground count 0 |
| `MASK_MISSING` | WARN | No `*_mask.nii.gz` beside the series; the image can still be exported. | absent |

## Engine-level findings

These are emitted by `run_qc` when the rule registry cannot inspect a healthy
volume, and are included in the same report:

| Code | Severity | Detects |
|---|---|---|
| `LOAD_FAILED` | FAIL | Volume reconstruction raised `GeometryError` (no readable slices, missing geometry tags, inconsistent array shapes). |
| `MASK_UNREADABLE` | FAIL | The mask file exists but SimpleITK cannot read it as a 3D volume. |
| `MASK_NOT_BINARY` | WARN | The mask contains labels outside `{0, 1}`; nearest-neighbour resampling preserves labels but binary Dice is not meaningful. |

## Why these rules

- **Geometry before pixels.** A model trained on a silently duplicated or
  gapped stack learns the acquisition error. Geometry failures are hard gates.
- **Quarantine, not repair.** The pipeline never invents missing slices. A
  defective series is excluded and recorded in `reports/qc_report.csv` so the
  data owner can fix the source archive and re-run.
- **One tolerance source.** `GeometryTolerances` is shared by
  `validate_geometry` and the QC registry, so the ingest view and the QC verdict
  cannot drift apart.
- **Masks are checked as data, not decoration.** Grid mismatch and empty masks
  are failures; a missing mask is a warning because image-only export is a
  legitimate use case.
- **Warnings remain visible.** WARN series are exported but their verdict and
  findings are written into every manifest record, preserving auditability.

## Reproducing a report

```bash
dvqc qc --input data/fixtures/deid --reports reports
cat reports/qc_report.csv
```

The JSON report also carries the rule catalog, aggregate counts, per-rule
quarantine counts, and the configuration hash used for the run.
