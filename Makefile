.PHONY: mypy ty lint precommit-install precommit-run smoke mlflow-ui web-dev test coverage test-scripts test-all
PYRUN = uv run 

SRC ?= src/ev_load_fc
TEST_SCRIPT ?= character_analysis
TEST_SCRIPT_IMPORT ?= model_builder.main

help:
	@echo "Targets:"
	@echo "  make mypy"
	@echo "  make ty"
	@echo "  make lint"
	@echo "  make test"
	@echo "  make coverage"
	@echo "  make test-all"
	@echo "  make test-module"
	@echo "  make mlflow-ui"
	@echo "  make import-time"
   

mypy:
	$(PYRUN) mypy $(SRC)

ty:
	$(PYRUN) ty check

ruff:
	$(PYRUN) ruff check $(SRC)

lint: ruff mypy ty

bandit:
	$(PYRUN) bandit -r src/model_builder -ll -ii
safety:
	$(PYRUN) safety check --full-report

license_check:
	$(PYRUN) pip-licenses
	$(PYRUN) pip-licenses --fail-on "GPL" --partial-match

# Tests
coverage:
	@echo "Coverage target is disabled because no automated pytest suites are configured."


test-all: test-scripts
	$(PYRUN) pytest tests/ -v

test-module:
	$(PYRUN) python tests/test_module.py --module $(TEST_MODULE)


# Logging and auditing

mlflow-ui:
	$(PYRUN) mlflow ui --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

import-time:
	$(PYRUN) python -X importtime -m ${TEST_FILE_IMPORT} --help > logs/importtime.txt 2>&1