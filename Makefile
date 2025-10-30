VENV := .venv
PYTHON := python3
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: help venv install-dev test run-embed run-search run-analyze clean

help:
	@echo "Makefile targets for embedlab"
	@echo "  make venv              - create virtualenv at $(VENV)"
	@echo "  make install-dev       - install dev deps from requirements-dev.txt"
	@echo "  make test              - run pytest in the venv"
	@echo "  make run-embed IMAGES_DIR=... OUT=...    - run embed command"
	@echo "  make run-search INDEX=... QUERY_DIR=... [K=5] [JSON=1] - run search"
	@echo "  make run-analyze INDEX=... DUP_THRESHOLD=0.92 ANOMALY_TOP=8 [JSON=1] - run analyze"
	@echo "  make clean             - remove $(VENV) and python artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel

install-dev: venv
	$(PIP) install -r requirements-dev.txt

test: install-dev
	$(PYTEST) -q

run-embed: install-dev
	@if [ -z "$(IMAGES_DIR)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make run-embed IMAGES_DIR=... OUT=..."; exit 1; \
	fi
	$(PY) embedlab.py embed --images-dir "$(IMAGES_DIR)" --out "$(OUT)"

run-search: install-dev
	@if [ -z "$(INDEX)" ] || [ -z "$(QUERY_DIR)" ]; then \
		echo "Usage: make run-search INDEX=... QUERY_DIR=... [K=5] [JSON=1]"; exit 1; \
	fi
	FLAGS=""; \
	if [ -n "$(JSON)" ]; then FLAGS="--json"; fi; \
	$(PY) embedlab.py search --index "$(INDEX)" --query-dir "$(QUERY_DIR)" --k ${K:=5} $$FLAGS

run-analyze: install-dev
	@if [ -z "$(INDEX)" ]; then \
		echo "Usage: make run-analyze INDEX=... DUP_THRESHOLD=0.92 ANOMALY_TOP=8 [JSON=1]"; exit 1; \
	fi
	FLAGS=""; \
	if [ -n "$(JSON)" ]; then FLAGS="--json"; fi; \
	$(PY) embedlab.py analyze --index "$(INDEX)" --dup-threshold ${DUP_THRESHOLD:=0.92} --anomaly-top ${ANOMALY_TOP:=8} $$FLAGS

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
