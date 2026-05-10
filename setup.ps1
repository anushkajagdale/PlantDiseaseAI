# Plant Disease AI - Automated Setup Script for Windows
# Run this script to automatically set up the project
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "`n=== Plant Disease AI - Windows Setup ===" -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Current Python: $pythonVersion" -ForegroundColor Green

# Check if Python 3.11+
if ($pythonVersion -match "Python 3\.11|Python 3\.12|Python 3\.13") {
    Write-Host "✓ Python version compatible" -ForegroundColor Green
} else {
    Write-Host "⚠ WARNING: Python 3.11-3.13 recommended. You have: $pythonVersion" -ForegroundColor Yellow
    Write-Host "  TensorFlow may not work with Python 3.14+" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install requirements
Write-Host "`nInstalling dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n✗ Dependency installation failed!" -ForegroundColor Red
    Write-Host "Try running: pip install -r requirements.txt --force-reinstall" -ForegroundColor Yellow
    exit 1
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python test_setup.py

# Run the application
Write-Host "`nSetup complete! Starting Streamlit application..." -ForegroundColor Cyan
Write-Host "Opening http://localhost:8501 in your browser..." -ForegroundColor Green
Start-Sleep -Seconds 2
streamlit run app.py
