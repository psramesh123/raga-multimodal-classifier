PY ?= python
RAGA_DATA_DIR ?= data/RaagaDhvani-A novel curated Augmented Carnatic mus/RaagaDhvani_Dataset_VidushiArchana
export RAGA_DATA_DIR

.PHONY: install lint format test manifest train ablations

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

test:
	$(PY) -m pytest

manifest:
	$(PY) -m raga.cli manifest

# CV only. The sealed test set is scored only with an explicit:
#   python -m raga.cli train --config configs/baseline.yaml --evaluate-test
train:
	$(PY) -m raga.cli train --config configs/baseline.yaml

ablations:
	for c in configs/random_forest.yaml configs/hist_gb.yaml configs/ablations/*.yaml; do \
		$(PY) -m raga.cli train --config $$c || exit 1; \
	done
