.PHONY: \
	help \
	mypy \
	ty \
	ruff \
	lint \
	bandit \
	safety \
	license_check \
	extraction \
	enrichment \
	features \
	training \
	inference \
	streamlit-app \
	coverage \
	test-all \
	test-module \
	mlflow-ui \
	import-time
	
PYRUN = uv run

SRC ?= src/ev_load_fc
TEST_SCRIPT ?= character_analysis
TEST_SCRIPT_IMPORT ?= model_builder.main

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Code quality:"
	@echo "  mypy            Run mypy type checking"
	@echo "  ty              Run ty type checker"
	@echo "  ruff            Run ruff linter"
	@echo "  lint            Run ruff + mypy + ty"
	@echo "  bandit          Run bandit security scanner"
	@echo "  safety          Run safety dependency audit"
	@echo "  license_check   Check dependency licenses"
	@echo ""
	@echo "Pipeline scripts:"
	@echo "  extraction      Run data extraction script"
	@echo "  enrichment      Run data enrichment script"
	@echo "  features        Run feature engineering script"
	@echo "  training        Run model training script"
	@echo "  inference       Run inference script"
	@echo ""
	@echo "App:"
	@echo "  streamlit-app   Launch Streamlit portfolio app"
	@echo ""
	@echo "Tests:"
	@echo "  test-all        Run all pytest tests"
	@echo "  test-module     Run a single module test (TEST_MODULE=<name>)"
	@echo "  coverage        (disabled - no automated pytest suites configured)"
	@echo ""
	@echo "Logging & auditing:"
	@echo "  mlflow-ui       Launch MLflow UI at http://127.0.0.1:5001"
	@echo "  import-time     Profile import time (TEST_FILE_IMPORT=<module>)"
	@echo ""


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


# Scripts
extraction:
	$(PYRUN) scripts/run_extraction.py
enrichment:
	$(PYRUN) scripts/run_enrichment.py
features:
	$(PYRUN) scripts/run_features.py
training:
	$(PYRUN) scripts/run_training.py
inference:
	$(PYRUN) scripts/run_inference.py

# App
streamlit-app:
	streamlit run streamlit_app/Home.py

# Tests
coverage:
	@echo "Coverage target is disabled because no automated pytest suites are configured."


test-all:
	$(PYRUN) pytest tests/ -v

test-module:
	$(PYRUN) python tests/test_module.py --module $(TEST_MODULE)


# Logging and auditing

mlflow-ui:
	$(PYRUN) mlflow ui --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

import-time:
	$(PYRUN) python -X importtime -m ${TEST_FILE_IMPORT} --help > logs/importtime.txt 2>&1