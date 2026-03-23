#!/bin/bash
# Setup script for GO Benchmark Analysis Pipeline

set -e  # Exit on error

echo "=========================================="
echo "GO Benchmark Analysis Pipeline Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check for conda
if command -v conda &> /dev/null; then
    echo "✓ Found conda"
    HAS_CONDA=true
else
    echo "  conda not found (optional)"
    HAS_CONDA=false
fi

echo ""
echo "Choose installation method:"
echo "1) conda (recommended if available)"
echo "2) pip + venv"
echo "3) Skip dependency installation"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        if [ "$HAS_CONDA" = true ]; then
            echo ""
            echo "Creating conda environment..."
            conda env create -f environment.yml
            echo ""
            echo "✓ Environment created!"
            echo ""
            echo "Activate with: conda activate go_benchmark_analysis"
        else
            echo "❌ conda not available, please choose option 2"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✓ Virtual environment created"
        
        echo ""
        echo "Activating virtual environment..."
        source venv/bin/activate
        
        echo ""
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "✓ Dependencies installed"
        
        echo ""
        echo "Activate later with: source venv/bin/activate"
        ;;
    3)
        echo ""
        echo "Skipping dependency installation"
        echo "Make sure you have the required packages:"
        cat requirements.txt
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/figures
mkdir -p results/reports
mkdir -p results/data
echo "✓ Directories created"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your TSV files in data/raw/"
echo "2. Run: make test     (to verify with demo data)"
echo "3. Run: make all      (to process your data)"
echo ""
echo "For help: make help"
echo ""
