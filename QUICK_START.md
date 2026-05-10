# 🚀 Quick Start Guide - Plant Disease AI

Get your Plant Disease AI application running in 5 minutes!

## Prerequisites
- Python 3.11 (⚠️ NOT Python 3.14 - TensorFlow not compatible yet)
- pip package manager
- 2 GB free disk space

## Step 1: Check Python Version
```bash
python --version
# Should show: Python 3.11.x
```

❌ **Got Python 3.14 or higher?**
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- OR use conda: `conda create -n plants python=3.11`

## Step 2: Create Virtual Environment
```bash
# Navigate to project directory
cd "c:\plant disease\PlantDiseaseAI"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies (takes 5-10 minutes)
pip install -r requirements.txt
```

## Step 4: Verify Installation
```bash
# Quick test
python test_setup.py
```

**Expected output:**
```
✓ Python Version
✓ Required Packages
✓ Model File
✓ Dataset Structure
✓ Model Loading
✓ Utils Module
✓ Streamlit

Passed: 7/7
🎉 All checks passed!
```

## Step 5: Run the Application
```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open http://localhost:8501 in your browser!

## Step 6: Test the Application
1. Click "Choose an image..."
2. Select a plant leaf image (JPG, PNG, or JPEG)
3. View the classification result!

---

## ✅ Success Checklist

- [x] Python 3.11 installed
- [x] Virtual environment created and activated
- [x] Dependencies installed (no red errors)
- [x] test_setup.py passed all checks
- [x] Streamlit app running on http://localhost:8501
- [x] Successfully uploaded and classified an image

---

## 🆘 Common Issues & Solutions

### "python not found" or "python: command not found"
```bash
# Use python3 instead
python3 --version
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Make sure virtual environment is activated!
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall packages
pip install -r requirements.txt --force-reinstall
```

### "CUDA not available" or "GPU not detected"
This is normal! The app works on CPU. No action needed.

### "Streamlit port already in use"
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### "Out of memory" or app crashes
```bash
# Restart Python and clear cache
python -c "import tensorflow as tf; tf.keras.backend.clear_session()"
```

---

## 📊 Next Steps

### Local Development
- Modify `app.py` to customize the UI
- Edit `train_model.py` to retrain with new data
- Adjust hyperparameters in `train_model.py`

### Deploy to Production

#### Option 1: Deploy to Render (Easiest)
```bash
git init
git add .
git commit -m "Initial commit"
git push
# Then deploy from render.com (see SETUP_AND_DEPLOYMENT.md)
```

#### Option 2: Deploy to Railway
```bash
npm install -g @railway/cli
railway login
railway up
```

#### Option 3: Deploy with Docker
```bash
docker build -t plant-disease-ai .
docker run -p 8501:8501 plant-disease-ai
```

---

## 📚 Full Documentation

For detailed information, see:
- [SETUP_AND_DEPLOYMENT.md](./SETUP_AND_DEPLOYMENT.md) - Complete setup guide
- [README.md](./README.md) - Project overview
- [app.py](./app.py) - Application code
- [train_model.py](./train_model.py) - Model training

---

## 🎯 What to Do Next

1. ✅ Run locally and test
2. Test with different plant images
3. (Optional) Retrain model with your own data
4. Deploy to production
5. Share with others!

---

**Having issues?**
1. Run `python test_setup.py` and check output
2. Check [SETUP_AND_DEPLOYMENT.md](./SETUP_AND_DEPLOYMENT.md) troubleshooting section
3. Verify Python version: `python --version`
4. Make sure virtual environment is activated

**Stuck?** Check the terminal output carefully - error messages usually point to the solution! 🔍

---

**Enjoy classifying plant diseases! 🌿**
