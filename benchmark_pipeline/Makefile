# Makefile for GO Benchmark Analysis Pipeline
# This automates the entire analysis workflow

# Configuration
PYTHON := python3
INPUT_DIR := data/raw
OUTPUT_DIR := results
COMBINED_DATA := data/processed/combined_benchmark_data.tsv
FILE_PATTERN := *.tsv

# Directories
RAW_DATA_DIR := data/raw
PROCESSED_DATA_DIR := data/processed
RESULTS_DIR := results
FIGURES_DIR := $(RESULTS_DIR)/figures
REPORTS_DIR := $(RESULTS_DIR)/reports

# Phony targets (not actual files)
.PHONY: all setup preprocess analyze clean help test

# Default target
all: setup preprocess analyze
	@echo ""
	@echo "======================================"
	@echo "PIPELINE COMPLETE!"
	@echo "======================================"
	@echo "Results available in: $(RESULTS_DIR)/"
	@echo ""

# Help message
help:
	@echo "GO Benchmark Analysis Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  make all         - Run complete pipeline (setup + preprocess + analyze)"
	@echo "  make setup       - Create directory structure and check dependencies"
	@echo "  make preprocess  - Concatenate TSV files from multiple GO levels"
	@echo "  make analyze     - Run benchmark analysis"
	@echo "  make test        - Run with demo/sample data"
	@echo "  make clean       - Remove generated files and results"
	@echo "  make help        - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  INPUT_DIR    = $(INPUT_DIR)"
	@echo "  OUTPUT_DIR   = $(OUTPUT_DIR)"
	@echo "  FILE_PATTERN = $(FILE_PATTERN)"
	@echo ""

# Setup: Create directory structure and check dependencies
setup:
	@echo "======================================"
	@echo "SETTING UP ENVIRONMENT"
	@echo "======================================"
	@mkdir -p $(RAW_DATA_DIR)
	@mkdir -p $(PROCESSED_DATA_DIR)
	@mkdir -p $(FIGURES_DIR)
	@mkdir -p $(REPORTS_DIR)
	@echo "✓ Directory structure created"
	@echo ""
	@echo "Checking Python dependencies..."
	@$(PYTHON) -c "import pandas, numpy, matplotlib, seaborn, scipy" 2>/dev/null || \
		(echo "❌ Missing dependencies. Install with: pip install -r requirements.txt" && exit 1)
	@echo "✓ All dependencies installed"
	@echo ""

# Preprocess: Concatenate multiple TSV files
preprocess: setup
	@echo "======================================"
	@echo "PREPROCESSING DATA"
	@echo "======================================"
	@if [ ! -d "$(INPUT_DIR)" ]; then \
		echo "❌ Input directory not found: $(INPUT_DIR)"; \
		echo "Please create it and add your TSV files"; \
		exit 1; \
	fi
	@if [ -z "$$(ls -A $(INPUT_DIR)/$(FILE_PATTERN) 2>/dev/null)" ]; then \
		echo "❌ No files matching '$(FILE_PATTERN)' found in $(INPUT_DIR)"; \
		echo "Please add your TSV files from different GO levels"; \
		exit 1; \
	fi
	$(PYTHON) preprocess.py $(INPUT_DIR) \
		--output $(COMBINED_DATA) \
		--pattern "$(FILE_PATTERN)"
	@echo ""

# Analyze: Run benchmark analysis on preprocessed data
analyze: preprocess
	@echo "======================================"
	@echo "RUNNING BENCHMARK ANALYSIS"
	@echo "======================================"
	@if [ ! -f "$(COMBINED_DATA)" ]; then \
		echo "❌ Preprocessed data not found: $(COMBINED_DATA)"; \
		echo "Run 'make preprocess' first"; \
		exit 1; \
	fi
	$(PYTHON) run_pipeline.py \
		--input $(COMBINED_DATA) \
		--output $(RESULTS_DIR)
	@echo ""

# Test: Run pipeline with demo data
test: setup
	@echo "======================================"
	@echo "RUNNING TEST WITH DEMO DATA"
	@echo "======================================"
	$(PYTHON) demo.py
	@echo ""
	@echo "✓ Demo completed successfully!"
	@echo "Check demo_output/ for results"
	@echo ""

# Clean: Remove all generated files
clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(PROCESSED_DATA_DIR)
	@rm -rf $(RESULTS_DIR)
	@rm -rf demo_output
	@rm -rf __pycache__
	@rm -f *.pyc
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -delete
	@echo "✓ Cleanup complete"

# Deep clean: Also remove setup directories
distclean: clean
	@echo "Deep cleaning..."
	@rm -rf data/
	@rm -rf venv/
	@rm -rf .pytest_cache/
	@echo "✓ Deep clean complete"
