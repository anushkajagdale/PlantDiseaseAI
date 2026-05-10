#!/bin/bash

# Plant Disease AI - Automated Setup Script for macOS/Linux
# Run this script to automatically set up the project
# Usage: bash setup.sh

echo ""
echo "=== Plant Disease AI - macOS/Linux Setup ===" 
echo ""

# Check Python version
echo "Checking Python installation..."
python_version=$(python3 --version 2>&1)
echo "Current Python: $python_version"

# Check if Python 3.11+
if [[ $python_version =~ "Python 3.11" ]] || [[ $python_version =~ "Python 3.12" ]] || [[ $python_version =~ "Python 3.13" ]]; then
    echo "✓ Python version compatible"
else
    echo "⚠ WARNING: Python 3.11-3.13 recommended. You have: $python_version"
    echo "  TensorFlow may not work with Python 3.14+"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Dependencies installed successfully!"
else
    echo ""
    echo "✗ Dependency installation failed!"
    echo "Try running: pip install -r requirements.txt --force-reinstall"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
python test_setup.py

# Run the application
echo ""
echo "Setup complete! Starting Streamlit application..."
echo "Opening http://localhost:8501 in your browser..."
sleep 2
streamlit run app.py
