.PHONY: install test benchmark benchmark-quick plots report docker-build docker-run clean lint coverage all

PYTHON := python3
PIP := pip
PYTEST := pytest
DOCKER_IMAGE := gdsf-gptcache:latest

# Install all dependencies
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Run unit tests
test:
	$(PYTEST) tests/ -v --tb=short

# Run full benchmark suite (all cache sizes, all policies, 30 runs)
benchmark:
	$(PYTHON) -m benchmarks.run_all --n-runs 30 --output-dir results/benchmarks

# Run quick benchmark (single cache size, 3 runs)
benchmark-quick:
	$(PYTHON) -m benchmarks.run_all --quick

# Generate all plots from benchmark results
plots:
	$(PYTHON) scripts/generate_plots.py

# Generate LaTeX report (requires results)
report: plots
	@echo "Results and plots generated in results/"

# Build Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run benchmarks in Docker
docker-run:
	docker compose up --build

# Clean generated files
clean:
	rm -rf results/plots/*.png results/plots/*.pdf
	rm -rf results/benchmarks/*.csv results/ablation/*.csv
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf src/__pycache__ tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run linter
lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/
	$(PYTHON) -m mypy src/ --ignore-missing-imports

# Run tests with coverage report
coverage:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing

# Run everything: lint, test, benchmark, plots
all: lint test benchmark plots
	@echo "All tasks completed successfully."
