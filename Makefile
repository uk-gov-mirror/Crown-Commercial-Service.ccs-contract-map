# Makefile for CCS Contract Map
# Provides common development commands

.PHONY: help install dev-install test lint format type-check security docker-build docker-run clean setup calibrator

# Default target
help:
	@echo "CCS Contract Map - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         - Install production dependencies"
	@echo "  make dev-install     - Install with dev dependencies"
	@echo "  make setup           - Full setup (venv + deps + pre-commit)"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run unit tests with coverage"
	@echo "  make test-watch      - Run tests in watch mode"
	@echo "  make lint            - Run all linters (ruff, mypy, bandit)"
	@echo "  make format          - Auto-format code with ruff"
	@echo "  make type-check      - Run mypy type checking"
	@echo "  make security        - Run bandit security scan"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run API server in Docker"
	@echo "  make docker-compose-up - Start all services with docker-compose"
	@echo ""
	@echo "ML/Calibration:"
	@echo "  make calibrator      - Build confidence calibrator from eval results"
	@echo "  make eval            - Run evaluation pipeline (requires MLflow)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           - Clean build artifacts and caches"
	@echo "  make check           - Run all checks (lint + test)"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

setup:
	python start.py --setup

# =============================================================================
# Development
# =============================================================================

test:
	pytest -v --cov=src/contractmap --cov-report=term-missing

test-watch:
	ptw --runner="pytest -v --cov=src/contractmap --cov-report=term-missing"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

type-check:
	mypy src/contractmap

security:
	bandit -r src/ -ll

check: lint type-check test

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t ccs-contract-map:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env ccs-contract-map:latest

docker-compose-up:
	docker compose up -d

docker-compose-down:
	docker compose down

docker-logs:
	docker compose logs -f api

# =============================================================================
# ML/Calibration
# =============================================================================

calibrator:
	python scripts/build_calibrator.py --results-dir data/results --output calibrator.pkl

eval:
	python eval/run_eval.py --help

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Pre-commit
# =============================================================================

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files