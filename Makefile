PY ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

.PHONY: install lint test fixtures pipeline clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

fixtures:
	$(PY) -c "from dicomvolumeqc.fixtures import build_fixture_set; arts = build_fixture_set('data/fixtures'); print(f'wrote {len(arts)} series for {len({a.patient_id for a in arts})} patients to data/fixtures')"

pipeline:
	$(PY) -m dicomvolumeqc.cli run-all --out data/fixtures --reports reports --export export

clean:
	rm -rf data/fixtures export .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
