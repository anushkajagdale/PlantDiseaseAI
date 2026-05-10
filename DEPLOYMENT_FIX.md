# Cloud Deployment Fix Guide

## Issues Fixed ✅

### 1. **Model Misclassification** (Plant Disease Detection)
- ✅ Added MobileNetV2 preprocessing to prediction pipeline
- **Effect**: Healthy leaves now classified correctly instead of showing diseased
- **Files modified**: `utils.py`

### 2. **403 Error on Image Upload** (Cloud Deployment)
- ✅ Improved error handling for file upload/temporary file creation
- ✅ Updated `.gitignore` to ensure model file is deployed
- ✅ Updated `.dockerignore` to keep model files
- ✅ Added better diagnostics during model loading
- **Files modified**: `app.py`, `.gitignore`, `.dockerignore`, `utils.py`

---

## Steps to Redeploy

### **Option 1: Render.com Deployment**

1. **Push code changes to GitHub**:
   ```bash
   git add -A
   git commit -m "Fix: Add MobileNetV2 preprocessing and improve error handling"
   git push origin main
   ```

2. **Trigger redeploy in Render Dashboard**:
   - Go to your Render service dashboard
   - Click **"Manual Deploy"** → **"Deploy latest commit"**
   - Wait for build to complete (5-10 minutes)

3. **Test the deployed app**:
   - Visit your Render URL
   - Upload a healthy leaf image
   - Verify it shows **"Healthy"** classification

---

### **Option 2: Railway.app Deployment**

1. **Push code changes to GitHub**:
   ```bash
   git add -A
   git commit -m "Fix: Add MobileNetV2 preprocessing and improve error handling"
   git push origin main
   ```

2. **Trigger redeploy in Railway**:
   - Railway auto-deploys on push, or manually trigger from dashboard
   - Monitor build logs for errors

3. **Test the deployed app**:
   - Visit your Railway service URL
   - Upload a healthy leaf image
   - Check the deployment logs for model loading messages

---

### **Option 3: Docker Build (Local Testing)**

To test before deploying to cloud:

```bash
# Build Docker image locally
docker build -t plant-disease-ai .

# Run container
docker run -p 8501:8501 plant-disease-ai

# Open browser to http://localhost:8501
```

---

## Debugging the 403 Error

If you still see 403 errors after redeployment:

1. **Check deployment logs** for model loading messages:
   - Should show: ✅ Model loaded successfully! (Size: X.X MB)
   - If missing, model file isn't deploying

2. **Verify file permissions**:
   - The app now catches and reports PermissionError
   - Check the error message in the Streamlit UI

3. **Common causes**:
   - **Read-only filesystem**: Some cloud providers have read-only `/tmp`
   - **Missing model file**: Ensure `model/plant_disease_model.keras` exists locally
   - **Disk space**: Large model (23.8 MB) may exceed free tier limits

---

## What Changed

### `utils.py`
- Added `preprocess_input` from `mobilenet_v2`
- Now applies proper preprocessing: `img_array = preprocess_input(img_array)`
- Added better logging for model loading

### `app.py`
- Better error handling for file upload
- Detects and reports PermissionError separately
- Verifies temp file is created and not empty
- Better cleanup on error

### `.gitignore`
- Added `!model/*.keras` to ensure model is included in Git

### `.dockerignore`
- Added note to prevent accidentally excluding model files

---

## Next Steps

1. **Verify model works locally**:
   ```bash
   streamlit run app.py
   ```
   - Test with both healthy and diseased leaf images

2. **Push to GitHub and redeploy to cloud**

3. **Monitor deployment logs** for any errors

4. **Report any remaining 403 errors** with:
   - Full error message from browser console
   - Deployment logs (Render/Railway dashboard)
   - Screenshot of the error

---

## Support

If the 403 error persists after redeployment, it likely indicates:
- **Cloud storage/permission issue** (beyond model scope)
- **Streamlit version compatibility** on the cloud platform
- **Temporary file system restrictions** on the cloud server

Contact support with deployment logs for further investigation.
