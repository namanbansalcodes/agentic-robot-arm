PY := .venv/bin/python
PYTEST := .venv/bin/pytest
UV := uv

# pybullet's bundled zlib does `#define fdopen(fd,mode) NULL`, which collides with
# the macOS SDK declaration of fdopen. Overriding the macro is what makes it build.
BUILD_ENV := CFLAGS="-std=gnu17 -Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"

.PHONY: help setup spike test one-shot agentic judge judge-live record report evidence clean

# A bare `make` must never rebuild the venv by accident.
.DEFAULT_GOAL := help

help:
	@echo "make setup       create .venv and install pinned deps"
	@echo "make spike       run the feasibility spike"
	@echo "make test        run the test suite (incl. the ground-truth firewall)"
	@echo "make judge       THE headline target: full eval offline from cache, then report"
	@echo "make judge-live  same eval, but calls the live VLM (needs SECRETS)"
	@echo "make record      record just some episodes live: make record SCENES=disturb_match3 CONDITIONS=agentic"
	@echo "make one-shot    run only the blind open-loop condition"
	@echo "make agentic     run only the self-verifying agentic condition"
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

one-shot:
	$(PY) -m harness.run --conditions one_shot --mode replay

agentic:
	$(PY) -m harness.run --conditions agentic --mode replay

# The headline target. Runs the ENTIRE eval offline, free, from the committed cache.
judge: test
	$(PY) -m harness.run --conditions all --mode replay
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

# Live mode needs GEMINI_API_KEY in the ENVIRONMENT, not just in the SECRETS file --
# sourcing it here is what makes the documented `cp SECRETS.example SECRETS` flow
# actually work. Without this the target dies with "GEMINI_API_KEY is not set".
judge-live:
	@test -f SECRETS || { echo "SECRETS not found. cp SECRETS.example SECRETS and put your key in it."; exit 1; }
	set -a; . ./SECRETS; set +a; $(PY) -m harness.run --conditions all --mode live
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

# Record only the episodes the replay cache is missing. Scoped, so it costs cents
# rather than re-paying for the whole eval. CONDITIONS/SCENES/SEEDS are overridable:
#   make record CONDITIONS=agentic SCENES=disturb_match3
CONDITIONS ?= all
SCENES ?= all
SEEDS ?= 0,1,2,3,4
record:
	@test -f SECRETS || { echo "SECRETS not found. cp SECRETS.example SECRETS and put your key in it."; exit 1; }
	set -a; . ./SECRETS; set +a; $(PY) -m harness.run --conditions $(CONDITIONS) \
		--scenes $(SCENES) --seeds $(SEEDS) --mode live --out results/record

report:
	$(PY) -m harness.report

evidence:
	$(PY) -m harness.report --evidence
	@echo "curated evidence written to docs/evidence/"

clean:
	rm -rf results/* spike/out/*
