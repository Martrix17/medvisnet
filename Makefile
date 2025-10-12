.PHONY: build run lint format test dev clean

# -------------------------------------------------------------------
# Docker / Devcontainer
# -------------------------------------------------------------------

IMAGE_NAME = medvisnet-dev
DOCKERFILE = .devcontainer/Dockerfile
CONTEXT = .

build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(CONTEXT)

run:
	docker run --gpus all -it --rm \
		--shm-size=8g \
		-v $(PWD):/workspace \
		$(IMAGE_NAME) bash

# -------------------------------------------------------------------
# Code Quality
# -------------------------------------------------------------------

lint:
	@echo "🔍 Running lint checks..."
	black --check --diff src tests
	isort --check-only src tests
	flake8 --color=always src tests --exclude __pycache__,.devcontainer

format:
	@echo "🧹 Auto-formatting code..."
	isort src tests
	black src tests

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

test:
	@echo "🧪 Running tests with coverage..."
	pytest -q --disable-warnings --cov=src --cov-report=term-missing --maxfail=1

# -------------------------------------------------------------------
# Local Testing
# -------------------------------------------------------------------

test-local:
	@echo "🧪 Running local tests..."
	pytest -q --disable-warnings --cov=src --cov-report=term-missing --maxfail=1

# -------------------------------------------------------------------
# Dev Utilities
# -------------------------------------------------------------------

dev:
	pip install -e . && pip install -r requirements.txt

clean:
	@echo "🗑️ Cleaning cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
