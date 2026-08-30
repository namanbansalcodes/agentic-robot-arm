PY := .venv/bin/python
PYTEST := .venv/bin/pytest
UV := uv

# pybullet's bundled zlib does `#define fdopen(fd,mode) NULL`, which collides with
# the macOS SDK declaration of fdopen. Overriding the macro is what makes it build.
BUILD_ENV := CFLAGS="-std=gnu17 -Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"

.PHONY: help setup spike test baseline agent judge judge-live report evidence clean

# A bare `make` must never rebuild the venv by accident.
.DEFAULT_GOAL := help

help:
	@echo "make setup       create .venv and install pinned deps"
	@echo "make spike       run the feasibility spike"
	@echo "make test        run the test suite (incl. the ground-truth firewall)"
	@echo "make judge       THE headline target: full eval offline from cache, then report"
	@echo "make judge-live  same eval, but calls the live VLM (needs SECRETS)"
	@echo "make baseline    run only the blind open-loop baseline condition"
	@echo "make agent       run only the self-verifying agent condition"
	@echo "make report      rebuild the report from existing results/"
	@echo "make evidence    curate the judge-facing pack into docs/evidence/"
	@echo "make clean       wipe results/ and spike/out/"

setup:
	$(UV) venv --python 3.11 .venv
	$(BUILD_ENV) VIRTUAL_ENV=.venv $(UV) pip install -r requirements.txt
	@echo "setup done. copy SECRETS.example -> SECRETS and add GEMINI_API_KEY for live runs."

spike:
	$(PY) spike/spike.py

test:
	$(PYTEST) tests/ -v

baseline:
	$(PY) -m harness.run --conditions baseline --mode replay

agent:
	$(PY) -m harness.run --conditions agent --mode replay

# The headline target. Runs the ENTIRE eval offline, free, from the committed cache.
judge: test
	$(PY) -m harness.run --conditions all --mode replay
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

judge-live:
	$(PY) -m harness.run --conditions all --mode live
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

report:
	$(PY) -m harness.report

evidence:
	$(PY) -m harness.report --evidence
	@echo "curated evidence written to docs/evidence/"

clean:
	rm -rf results/* spike/out/*
