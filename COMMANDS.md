# 🌿 Plant Disease AI - Command Reference & Setup Guide

## 📋 Complete Command Guide

### WINDOWS Users
```powershell
# AUTOMATED SETUP (Recommended)
powershell -ExecutionPolicy Bypass -File setup.ps1

# OR MANUAL SETUP
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python test_setup.py

# 6. Run application
streamlit run app.py

# 7. Open browser to http://localhost:8501
```

### macOS/Linux Users
```bash
# AUTOMATED SETUP (Recommended)
bash setup.sh

# OR MANUAL SETUP
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python test_setup.py

# 6. Run application
streamlit run app.py

# 7. Open browser to http://localhost:8501
```

---

## 🔍 Verification Commands

```bash
# Check Python version (must be 3.11+)
python --version

# Check if virtualenv is activated
# Windows: Should see (venv) in prompt
# macOS/Linux: Should see (venv) in prompt

# List installed packages
pip list

# Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Check Streamlit
python -c "import streamlit as st; print(f'Streamlit: {st.__version__}')"

# Run full system check
python test_setup.py

# Check model file
python -c "import os; print(f'Model exists: {os.path.exists(\"model/plant_disease_model.keras\")}')"

# Check dataset structure
python -c "import os; print(sorted(os.listdir('dataset/train')))"
```

---

## 🚀 Running the Application

### Start Streamlit App
```bash
# Default port 8501
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Headless mode (production)
streamlit run app.py --server.headless true
```

### Access Application
- Local: http://localhost:8501
- Network: http://<your-ip>:8501

### Stop Application
Press `Ctrl+C` in terminal

---

## 🧪 Testing Commands

### Verify Setup
```bash
python test_setup.py
```

### Test Model Loading
```bash
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('model/plant_disease_model.keras')
print('✓ Model loaded successfully!')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
"
```

### Test Predictions
```bash
python -c "
from utils import predict_image, CLASS_NAMES
import os
print(f'Classes: {CLASS_NAMES}')
# Predictions will work with actual image files
"
```

### Test Imports
```bash
python -c "
import streamlit; import tensorflow; import numpy
import PIL; import cv2; import sklearn
print('✓ All imports successful!')
"
```

---

## 📦 Dependency Management

### Update All Packages
```bash
pip install -r requirements.txt --upgrade
```

### Reinstall All Packages
```bash
pip install -r requirements.txt --force-reinstall
```

### Check for Conflicts
```bash
pip check
```

### Generate Requirements from Environment
```bash
pip freeze > requirements_current.txt
```

### Install Specific Package
```bash
pip install tensorflow==2.13.0
pip install streamlit==1.28.0
```

---

## 🐳 Docker Commands

### Build Docker Image
```bash
docker build -t plant-disease-ai .
```

### Run Docker Container
```bash
docker run -p 8501:8501 plant-disease-ai
```

### Run with Custom Port
```bash
docker run -p 9000:8501 plant-disease-ai
```

### View Docker Images
```bash
docker images
```

### Remove Docker Image
```bash
docker rmi plant-disease-ai
```

### Run Interactive Container
```bash
docker run -it -p 8501:8501 plant-disease-ai bash
```

---

## 📤 Deployment Commands

### Deploy to Render
```bash
# Push to GitHub first
git add .
git commit -m "Deploy to Render"
git push origin main

# Then on render.com:
# 1. Connect GitHub repo
# 2. Set environment variables
# 3. Deploy!
```

### Deploy to Railway
```bash
npm install -g @railway/cli
railway login
railway link
railway up
```

### Deploy to Docker Registry
```bash
# Build
docker build -t plant-disease-ai .

# Tag for Docker Hub
docker tag plant-disease-ai username/plant-disease-ai:latest

# Push to Docker Hub
docker push username/plant-disease-ai:latest
```

---

## 🔧 Environment Setup

### Create .env File
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

### Edit Environment Variables
```bash
# Windows (Notepad)
notepad .env

# macOS/Linux (vim)
vim .env

# VS Code
code .env
```

### Common Environment Variables
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
PYTHONUNBUFFERED=1
MODEL_PATH=model/plant_disease_model.keras
DATASET_PATH=dataset
```

---

## 🚨 Troubleshooting Commands

### Fix Python Path
```bash
# Windows - Check Python location
where python
where python3

# macOS/Linux - Check Python location
which python3
which python
```

### Check Virtual Environment
```bash
# Activate and verify
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Should show (venv) in prompt
```

### Clear Python Cache
```bash
# Clear __pycache__
python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"

# Clear TensorFlow cache
python -c "import tensorflow as tf; tf.keras.backend.clear_session()"
```

### Reinstall After Errors
```bash
# Remove virtual environment
rmdir venv /s /q  # Windows
rm -rf venv  # macOS/Linux

# Create fresh virtual environment
python -m venv venv

# Activate and reinstall
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Check Port Usage
```bash
# Windows
netstat -ano | findstr :8501

# macOS/Linux
lsof -i :8501
```

### Kill Process on Port
```bash
# Windows
taskkill /PID <PID> /F

# macOS/Linux
kill -9 $(lsof -t -i:8501)
```

---

## 📊 Model Training Commands

### Train Model
```bash
python train_model.py
```

### Organize Dataset
```bash
python organize_dataset.py
```

### Balance Dataset
```bash
python balance_healthy.py
```

### Full Training Pipeline
```bash
# 1. Organize data
python organize_dataset.py

# 2. Balance classes
python balance_healthy.py

# 3. Train model
python train_model.py

# 4. Verify model
streamlit run app.py
```

---

## 🔄 Git Commands

### Initialize Git
```bash
git init
git add .
git commit -m "Initial commit"
```

### Add Remote
```bash
git remote add origin https://github.com/username/PlantDiseaseAI.git
git push -u origin main
```

### Update Remote
```bash
git add .
git commit -m "Update message"
git push origin main
```

### Clone Repository
```bash
git clone https://github.com/username/PlantDiseaseAI.git
cd PlantDiseaseAI
```

---

## 📝 Useful Aliases (macOS/Linux)

```bash
# Add to ~/.bashrc or ~/.zshrc
alias activate_venv='source venv/bin/activate'
alias run_app='streamlit run app.py'
alias test_setup='python test_setup.py'
alias train='python train_model.py'
```

---

## 🎯 Complete Setup Script

### Windows (All-in-One)
```powershell
# Save as setup_complete.ps1
$ErrorActionPreference = "Stop"

Write-Host "🌿 Plant Disease AI - Complete Setup" -ForegroundColor Cyan

# Check Python
python --version

# Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python test_setup.py

# Run
streamlit run app.py
```

### macOS/Linux (All-in-One)
```bash
#!/bin/bash
set -e

echo "🌿 Plant Disease AI - Complete Setup"

# Check Python
python3 --version

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python test_setup.py

# Run
streamlit run app.py
```

---

## 📚 Quick Reference Table

| Task | Windows | macOS/Linux |
|------|---------|------------|
| Create venv | `python -m venv venv` | `python3 -m venv venv` |
| Activate venv | `venv\Scripts\activate` | `source venv/bin/activate` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` | `streamlit run app.py` |
| Stop app | `Ctrl+C` | `Ctrl+C` |
| Test setup | `python test_setup.py` | `python test_setup.py` |
| Deactivate venv | `deactivate` | `deactivate` |

---

## 💡 Pro Tips

### 1. Save Bandwidth (Use Cache)
```bash
pip install --cache-dir ./pip-cache -r requirements.txt
```

### 2. Parallel Installation
```bash
pip install --use-deprecated=legacy-resolver -r requirements.txt
```

### 3. Quiet Mode
```bash
pip install -q -r requirements.txt
streamlit run app.py --logger.level=error
```

### 4. Background Running (macOS/Linux)
```bash
nohup streamlit run app.py > app.log 2>&1 &
```

### 5. Background Running (Windows)
```powershell
Start-Process streamlit -ArgumentList "run app.py" -WindowStyle Hidden
```

---

## 🆘 Quick Fixes

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt --force-reinstall --no-cache-dir
```

### "Permission denied"
```bash
# Windows: Run as Administrator
# macOS/Linux: Use sudo (if needed)
sudo pip install -r requirements.txt
```

### "Port already in use"
```bash
streamlit run app.py --server.port 9000
```

### "Memory full"
```bash
# Clear cache
pip cache purge
# Or restart Python
python -c "import gc; gc.collect()"
```

### "SSL Certificate Error"
```bash
pip install --trusted-host pypi.python.org -r requirements.txt
```

---

## 📞 Getting Help

### Check Documentation
```bash
# View README
more README.md  # Windows
less README.md  # macOS/Linux

# View Setup Guide
more SETUP_AND_DEPLOYMENT.md
```

### Check Error Details
1. Read full error message
2. Run `python test_setup.py`
3. Check [SETUP_AND_DEPLOYMENT.md](./SETUP_AND_DEPLOYMENT.md#troubleshooting)
4. Try suggested fixes

### Check Python Version
```bash
python --version  # Must be 3.11+
python -c "import sys; print(sys.executable)"
```

---

## 🎯 Final Checklist

Execute these in order:

```bash
# 1. Verify Python ✓
python --version

# 2. Setup virtual environment ✓
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies ✓
pip install -r requirements.txt

# 4. Run verification ✓
python test_setup.py

# 5. Start application ✓
streamlit run app.py

# 6. Open browser ✓
# http://localhost:8501
```

---

## 📖 Documentation Files

- [README.md](./README.md) - Project overview
- [QUICK_START.md](./QUICK_START.md) - 5-minute setup
- [SETUP_AND_DEPLOYMENT.md](./SETUP_AND_DEPLOYMENT.md) - Complete guide
- [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) - Completion summary
- [COMMANDS.md](./COMMANDS.md) - This file!

---

**You're ready! Start with `powershell -ExecutionPolicy Bypass -File setup.ps1` (Windows) or `bash setup.sh` (macOS/Linux)** 🚀
