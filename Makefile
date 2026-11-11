PY ?= python

.PHONY: all install data experiments quick analyze figure test clean

all: experiments analyze figure

install:
	$(PY) -m pip install -r requirements.txt

data:            ## rebuild data/processed/students.csv from OULAD (already shipped)
	$(PY) scripts/01_prepare_data.py --source github

experiments:     ## full run of the paper (about 10 minutes on one CPU core)
	$(PY) scripts/02_run_experiments.py

quick:           ## 3 cohorts per setting, smoke test
	$(PY) scripts/02_run_experiments.py --quick --out results/raw/quick

analyze:
	$(PY) scripts/03_analyze.py

figure:
	$(PY) scripts/04_make_figure.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results/raw/parts results/raw/quick .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
