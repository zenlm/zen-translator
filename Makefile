# Zen Translator Makefile
# Real-time multimodal translation with voice cloning and lip sync

SHELL := /bin/bash
PYTHON := python3
UV := uv
VENV := .venv
MODEL_DIR := ./models

.PHONY: all install dev clean test lint format serve train download help

all: install download

## Installation

install: venv  ## Install production dependencies
	$(UV) pip install -e .

dev: venv  ## Install development dependencies
	$(UV) pip install -e ".[all]"
	$(UV) pip install git+https://github.com/huggingface/transformers

venv:  ## Create virtual environment
	$(UV) venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Activate with: source $(VENV)/bin/activate"

## Model Downloads

download: download-qwen3-omni download-cosyvoice download-wav2lip  ## Download all models

download-qwen3-omni:  ## Download Qwen3-Omni (30B)
	@echo "Downloading Qwen3-Omni-30B-A3B-Instruct..."
	$(UV) run hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir $(MODEL_DIR)/qwen3-omni

download-cosyvoice:  ## Download CosyVoice 2.0
	@echo "Downloading CosyVoice 2.0..."
	$(UV) run hf download FunAudioLLM/CosyVoice2-0.5B --local-dir $(MODEL_DIR)/cosyvoice

download-wav2lip:  ## Download Wav2Lip
	@echo "Downloading Wav2Lip..."
	$(UV) run hf download numz/wav2lip_studio --local-dir $(MODEL_DIR)/wav2lip

download-quantized:  ## Download quantized models (smaller)
	@echo "Downloading quantized Qwen3-Omni AWQ..."
	$(UV) run hf download cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit --local-dir $(MODEL_DIR)/qwen3-omni-4bit

## Running

serve:  ## Start the translation server
	$(UV) run zen-serve --host 0.0.0.0 --port 8000

serve-dev:  ## Start server with auto-reload
	$(UV) run zen-serve --host 0.0.0.0 --port 8000 --reload

translate:  ## Translate a file (use: make translate FILE=input.mp4)
	$(UV) run zen-translate $(FILE) -o output.mp4

## Training

train-identity:  ## Train Zen identity
	$(UV) run zen-translate train --type identity --output ./outputs/identity

train-anchor:  ## Train news anchor adaptation
	$(UV) run zen-translate train --type anchor --output ./outputs/anchor

dataset-build:  ## Build news anchor training dataset
	$(UV) run zen-translate dataset build --output ./data/news_anchors --channels cnn,bbc,nhk,dw

dataset-list:  ## List available news channels
	$(UV) run zen-translate dataset list

swift-train:  ## Run ms-swift training (after train-identity generates config)
	swift sft --config ./outputs/identity/train_config.yaml

## Development

test:  ## Run tests
	$(UV) run pytest tests/ -v --cov=zen_translator

lint:  ## Run linter
	$(UV) run ruff check src/ tests/

format:  ## Format code
	$(UV) run ruff format src/ tests/

typecheck:  ## Run type checker
	$(UV) run mypy src/

## Docker

docker-build:  ## Build Docker image
	docker build -t zenlm/zen-translator:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 --gpus all zenlm/zen-translator:latest

## Cleanup

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-models:  ## Remove downloaded models
	rm -rf $(MODEL_DIR)/*

clean-all: clean clean-models  ## Clean everything
	rm -rf $(VENV)

## Help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Default target
.DEFAULT_GOAL := help
