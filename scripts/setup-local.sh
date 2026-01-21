#!/bin/bash
# Seismic MLOps Pipeline - Local Setup
# Usage: ./scripts/setup-local.sh

set -e

echo "=============================================="
echo "Seismic MLOps Pipeline - Local Setup"
echo "=============================================="

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" < "3.11" ]]; then
    echo "WARNING: Python 3.11+ recommended. Current: $python_version"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/bronze data/silver data/gold
mkdir -p models mlruns feature_store

# Check Ollama (optional)
echo ""
echo "Checking Ollama (optional for LLM features)..."
if command -v ollama &> /dev/null; then
    echo "Ollama is installed."
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        echo "llama3.1:8b model is available."
    else
        echo "To enable LLM features, run: ollama pull llama3.1:8b"
    fi
else
    echo "Ollama not installed. LLM features will be disabled."
    echo "To install: https://ollama.ai/download"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python run_all_stages.py"
echo ""
echo "To run quick validation:"
echo "  python src/stage8_cicd.py"
