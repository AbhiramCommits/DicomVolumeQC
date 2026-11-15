# Contributing

This repository is a model of regulated-software practice, so contributions are
reviewed for reproducibility and traceability as much as for style.

## Workflow

1. **Branch.** Create a topic branch from `main`:
   `git switch -c feat/short-description`.
2. **Change one concern.** Keep code, tests, docs, and measured artifacts in the
   same change when they describe the same behavior.
3. **Run the gates locally.**
   ```bash
   make lint                      # ruff check .
   make test                      # pytest --cov --cov-fail-under=85
   make fixtures                  # regenerate the synthetic corpus
   dvqc run-all --out data/fixtures --reports reports --export export
   dvqc verify --export export
   ```
4. **Open a pull request.** Use `.github/pull_request_template.md`. Describe
   any change to QC thresholds, split ratios, de-identification behavior, or the
   manifest schema explicitly.
5. **CI.** GitHub Actions runs `ruff check .`, the test matrix (Python 3.11 and
   3.12) with the 85% coverage gate, then the end-to-end pipeline and manifest
   verification. CI must be green.
6. **Review.** At least one reviewer confirms the regulated-data invariants:
   no PHI leaks, patient-disjoint splits, quarantined series excluded, and all
   exported checksums verified.
7. **Merge.** Do not push generated data (`data/`, `export/`) or the
   de-identification crosswalk; they are intentionally ignored.

## Ground rules

- Never commit real PHI. The fixtures are synthetic and seeded; keep them that
  way. A new fixture defect must be injected by a named flag so QC has ground
  truth.
- Keep runs deterministic for a fixed config and seed. If a change makes output
  non-deterministic, call it out in the PR and in the manifest documentation.
- Update `README.md`, `docs/QC_RULES.md`, and `reports/metrics.json` from a real
  run whenever behavior or measured numbers change. Do not hand-edit invented
  figures.
- Public functions need type hints, modules should stay near 300 lines, and
  failures should raise the typed exceptions in `errors.py`.
- New QC rules must be registered with a code, severity, description, and tests
  proving the expected verdict on the injected fixture.

## Reproducing the reported numbers

```bash
.venv/bin/python -m pytest --cov=dicomvolumeqc \
  --cov-report=json:reports/coverage.json --cov-fail-under=85 -q
/usr/bin/time -p .venv/bin/python -m dicomvolumeqc.cli run-all \
  --out data/fixtures --reports reports --export export
# Write the measured real seconds (one line) to reports/final_wall_clock.txt.
.venv/bin/python scripts/record_metrics.py
```

The metrics script reads the pipeline summary, manifest, coverage JSON, and
measured wall-clock value; it does not invent numbers.
