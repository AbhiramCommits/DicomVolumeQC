# Pull request verification checklist

This repository models regulated-software (SaMD) practice, so every change must be
reviewable and reproducible.

- [ ] `ruff check .` is clean and `make test` passes locally with coverage ≥ 85%.
- [ ] `dvqc run-all` exits 0 and `dvqc verify --export export` reports all checksums OK.
- [ ] The generated `reports/qc_report.csv` was reviewed: new quarantine counts or rule
      changes are explained in the PR description, not silently accepted.
- [ ] `reports/deid_report.json` shows `leaks_after: 0`; no real or new PHI fixtures were added.
- [ ] Config changes (`configs/default.yaml`, split ratios, target spacing, tolerances) are
      called out with their effect on split sizes and exported shapes.
- [ ] The dataset manifest digest changed only for the reasons described above.
- [ ] Documentation/results that quote measured numbers were regenerated from this branch.

Reviewer: confirm the four regulated-data invariants — no PHI leaks, patient-disjoint
splits, quarantined series excluded from export, and every exported file checksum-verified.
