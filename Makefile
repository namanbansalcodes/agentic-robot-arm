PY := .venv/bin/python
PYTEST := .venv/bin/pytest
UV := uv

# pybullet's bundled zlib does `#define fdopen(fd,mode) NULL`, which collides with
# the macOS SDK declaration of fdopen. Overriding the macro is what makes it build.
BUILD_ENV := CFLAGS="-std=gnu17 -Dfdopen=fdopen" CXXFLAGS="-Dfdopen=fdopen"

.PHONY: setup spike test baseline agent judge judge-live report clean

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
judge:
	$(PY) -m harness.run --conditions all --mode replay
	$(PY) -m harness.report
	@echo "report: results/report.md  |  results/report.html"

judge-live:
	$(PY) -m harness.run --conditions all --mode live
	$(PY) -m harness.report

report:
	$(PY) -m harness.report

clean:
	rm -rf results/* spike/out/*
