# 🌿 Plant Disease AI - Complete Setup & Deployment Guide

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Local Setup](#local-setup)
3. [Running Locally](#running-locally)
4. [Testing the ML Pipeline](#testing-the-ml-pipeline)
5. [Deployment Options](#deployment-options)
6. [Production Optimization](#production-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Environment Variables](#environment-variables)

---

## Project Architecture

### Tech Stack
| Component | Technology | Version |
|-----------|------------|---------|
| Frontend | Streamlit | 1.28.0 |
| ML Model | TensorFlow | 2.13.0 |
| Model Type | MobileNetV2 Transfer Learning | - |
| Python | 3.11.9 (recommended) | - |
| Deployment | Docker, Railway, Render | - |

### Project Structure
```
PlantDiseaseAI/
├── app.py                          # Streamlit web interface
├── utils.py                        # Prediction utilities & model loading
├── train_model.py                  # Model training script
├── organize_dataset.py             # Dataset organization utility
├── balance_healthy.py              # Data augmentation script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── Procfile                        # Heroku deployment config
├── render.yaml                     # Render.com deployment config
├── railway.json                    # Railway.app deployment config
├── runtime.txt                     # Python version specification
├── .env.example                    # Environment variables template
├── .dockerignore                   # Docker build optimization
│
├── model/
│   └── plant_disease_model.keras   # Trained MobileNetV2 model (23.8 MB)
│
└── dataset/
    ├── train/
    │   ├── Healthy/               # Training images - Healthy leaves
    │   └── Diseased/              # Training images - Diseased leaves
    └── valid/
        ├── Healthy/               # Validation images - Healthy leaves
        └── Diseased/              # Validation images - Diseased leaves
```

### ML Pipeline Architecture
```
Input Image (224x224 RGB)
    ↓
Normalize (divide by 255)
    ↓
MobileNetV2 (pre-trained ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(2, Softmax) → [Healthy, Diseased]
    ↓
Output: Class Label + Confidence Score
```

---

## Local Setup

### Step 1: Install Python 3.11
**Why Python 3.11?** TensorFlow 2.13.0 requires Python 3.9-3.11. Python 3.14 is not yet supported.

**Windows:**
```bash
# Download from python.org and install Python 3.11.9
# During installation, check "Add Python to PATH"
python --version  # Should show: Python 3.11.9
```

**macOS:**
```bash
brew install python@3.11
python3.11 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### Step 2: Create Virtual Environment
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

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

**Installation Notes:**
- TensorFlow (2.13.0): ~500 MB, may take 5-10 minutes
- First installation downloads pre-trained weights
- Requires ~2 GB disk space total

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit as st; print(f'Streamlit version: {st.__version__}')"
```

---

## Running Locally

### Start the Application
```bash
# Make sure virtual environment is activated
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://<your-ip>:8501
```

### Access the Application
- Open browser: `http://localhost:8501`
- Upload an image (JPG, PNG, JPEG)
- View classification results

### Stop the Application
Press `Ctrl+C` in the terminal

---

## Testing the ML Pipeline

### Test 1: Verify Model Loading
```python
import tensorflow as tf
import os

model_path = "model/plant_disease_model.keras"
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded: {model_path}")
print(f"Model summary:\n{model.summary()}")
```

### Test 2: Test Prediction with Sample Image
```bash
# Create a test script: test_prediction.py
python test_prediction.py
```

### Test 3: Verify Dataset Structure
```bash
# Check dataset organization
python -c "import os; print(sorted(os.listdir('dataset/train')))"
# Expected output: ['Diseased', 'Healthy']
```

### Test 4: Check for Model Training
If you need to retrain the model:
```bash
python train_model.py
# This will:
# - Load MobileNetV2 pre-trained model
# - Phase 1: Train top layers (5 epochs)
# - Phase 2: Fine-tune full model (7 epochs)
# - Save updated model to model/plant_disease_model.keras
```

---

## Deployment Options

### Option 1: Render.com (Recommended for Beginners)

**Advantages:**
- Free tier: 750 hours/month (~31 days continuous)
- 0.5 CPU, 512 MB RAM
- GitHub integration
- Automatic deployments

**Steps:**
1. Push code to GitHub repository
2. Go to [render.com](https://render.com)
3. Click "New +"  → "Web Service"
4. Select GitHub repository
5. Configure:
   - **Name**: plant-disease-ai
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port 8501`
6. Deploy!

**Environment Variables:**
Add in Render dashboard under "Environment":
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONUNBUFFERED=1
```

### Option 2: Railway.app

**Advantages:**
- Free tier: $5/month credits (plenty for this app)
- Better performance than Render
- Simple CLI deployment

**Steps:**
1. Install Railway CLI: `npm install -g @railway/cli`
2. `railway login`
3. `railway init`
4. `railway up`
5. Visit your deployment URL

### Option 3: Hugging Face Spaces

**Advantages:**
- Optimized for Streamlit apps
- Free GPU available (optional)
- Model versioning built-in

**Steps:**
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Select "Streamlit" runtime
4. Upload files from GitHub or manually
5. Done! Auto-deploys on push

### Option 4: Docker Deployment (Advanced)

**Build locally:**
```bash
docker build -t plant-disease-ai .
docker run -p 8501:8501 plant-disease-ai
```

**Deploy to cloud:**
- AWS ECR + ECS
- DigitalOcean App Platform
- Google Cloud Run
- Azure Container Instances

---

## Production Optimization

### Memory & Performance Optimization

#### 1. Model Quantization (For Free Tier Hosting)
```python
# Convert model to TensorFlow Lite for smaller size
import tensorflow as tf

model = tf.keras.models.load_model('model/plant_disease_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model/plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Original: {os.path.getsize('model/plant_disease_model.keras') / 1024 / 1024:.2f} MB")
print(f"Optimized: {os.path.getsize('model/plant_disease_model.tflite') / 1024 / 1024:.2f} MB")
```

#### 2. Image Caching
Streamlit automatically caches model loading. Add to app.py:
```python
import streamlit as st

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/plant_disease_model.keras')
```

#### 3. Batch Processing
For higher throughput:
```python
# Process multiple images in one request
# Add to utils.py for bulk predictions
def predict_batch(image_paths):
    predictions = []
    for img_path in image_paths:
        label, confidence = predict_image(img_path)
        predictions.append((label, confidence))
    return predictions
```

### Database Integration (Optional)
For production with user history:
```bash
pip install sqlalchemy psycopg2-binary
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt --upgrade --force-reinstall
```

### ❌ "CUDA/GPU not available"
**Solution:**
TensorFlow will automatically fall back to CPU. For GPU support:
```bash
pip install tensorflow[and-cuda]
```

### ❌ "Model not found at model/plant_disease_model.keras"
**Solution:**
```bash
# Check if model exists
python -c "import os; print(os.path.exists('model/plant_disease_model.keras'))"
# If False, download from GitHub or retrain with train_model.py
```

### ❌ "Out of memory" on free tier hosting
**Solution:**
1. Use TensorFlow Lite quantized model
2. Add image preprocessing to reduce memory
3. Implement request queuing
4. Add memory-efficient Docker image

### ❌ "Streamlit app not starting on production"
**Solution:**
```bash
# Add to Procfile:
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --logger.level=error
```

### ⚠️ Slow Predictions (~5-10 seconds)
**Causes:**
- Running on shared CPU (free tier)
- Large model loading time
- Image preprocessing

**Solutions:**
1. Pre-load model at startup (✓ Already implemented)
2. Use TensorFlow Lite
3. Add predictions caching
4. Upgrade to paid tier

---

## Environment Variables

### Local Development (.env)
```bash
# Copy .env.example to .env
cp .env.example .env
```

**Contents:**
```env
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_CLIENT_LOGGER_LEVEL=info

# Model Configuration
MODEL_PATH=model/plant_disease_model.keras
DATASET_PATH=dataset

# Python Configuration
PYTHONUNBUFFERED=1
```

### Production Environment
Set in your hosting platform dashboard:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_CLIENT_LOGGER_LEVEL=error
PYTHONUNBUFFERED=1
```

---

## Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Development
streamlit run app.py

# Testing
python test_prediction.py
python train_model.py

# Deployment
docker build -t plant-disease-ai .
docker run -p 8501:8501 plant-disease-ai

# Debugging
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import streamlit as st; print(st.__version__)"
```

---

## Model Information

**Model**: MobileNetV2 + Custom Head
**Size**: 23.8 MB
**Input**: 224x224 RGB images
**Output**: 2 classes (Healthy, Diseased)
**Training Data**: Tomato plant images
**Accuracy**: ~95% (on validation set)
**Inference Time**: 0.5-1 second (CPU), ~100ms (GPU)

---

## Next Steps

1. ✅ Setup local environment
2. ✅ Test with sample images
3. ✅ Configure deployment platform
4. ✅ Deploy to production
5. 🔄 Monitor performance
6. 📊 Collect user feedback
7. 🚀 Add more plant species (future enhancement)

---

## Support & Documentation

- **TensorFlow**: https://www.tensorflow.org/guide
- **Streamlit**: https://docs.streamlit.io/
- **Render**: https://render.com/docs
- **Railway**: https://docs.railway.app/
- **Docker**: https://docs.docker.com/

---

## License

This project is provided as-is for educational and commercial use.

---

## Changelog

**v1.0 (Current)**
- ✅ Complete setup guide
- ✅ Deployment configurations
- ✅ Production optimizations
- ✅ Error handling & validation
- ✅ Environment variable management

---

**Last Updated**: May 8, 2026
**Status**: Production Ready ✅
