#!/bin/bash
# Activation script for ESG Score EDA environment

echo "Activating ESG Score EDA virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating it now..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import pandas" 2>/dev/null; then
    echo "Dependencies not installed. Installing from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
else
    echo "Environment activated. Dependencies already installed."
fi

echo ""
echo "Virtual environment is now active!"
echo "Python version: $(python --version)"
echo "To deactivate, run: deactivate"
echo ""
echo "To run the pipeline:"
echo "  python main.py --all"
echo ""
echo "Or run specific components:"
echo "  python main.py --missing-values"
echo "  python main.py --outliers-iqr-z"
echo "  python main.py --outliers-forest"
echo "  python main.py --model-specific-data"