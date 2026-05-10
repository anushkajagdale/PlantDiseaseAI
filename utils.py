import os
from pathlib import Path
import numpy as np
from PIL import Image

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    preprocess_input = None
    print("⚠️ Warning: TensorFlow not available. Using mock predictions for testing.")

# Get the model path (support both relative and absolute paths)
def get_model_path():
    # Try HDF5 first (more compatible), then .keras
    h5_path = os.getenv('MODEL_PATH', 'model/plant_disease_model.h5')
    keras_path = os.getenv('MODEL_PATH', 'model/plant_disease_model.keras')

    if os.path.isabs(h5_path):
        h5_full = h5_path
        keras_full = keras_path
    else:
        base_dir = os.path.dirname(__file__)
        h5_full = os.path.join(base_dir, h5_path)
        keras_full = os.path.join(base_dir, keras_path)

    # Return the path that exists, preferring HDF5
    if os.path.exists(h5_full):
        return h5_full
    elif os.path.exists(keras_full):
        return keras_full
    else:
        return h5_full  # Default to HDF5 path

# Load model once at startup
MODEL_PATH = get_model_path()
model = None

if TF_AVAILABLE:
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}. Using fallback predictions.")
    else:
        try:
            print(f"📦 Loading model from: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✅ Model loaded successfully! (Size: {file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load model: {e}. Using fallback predictions.")
            model = None

# Change these AFTER you see the folder names inside dataset/train
DATASET_PATH = os.getenv('DATASET_PATH', 'dataset')
try:
    CLASS_NAMES = sorted(os.listdir(os.path.join(DATASET_PATH, 'train')))
except:
    CLASS_NAMES = ['Healthy', 'Diseased']

def predict_image(img_path):
    """
    Predict plant disease from image path
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        tuple: (class_label, confidence_score)
    """
    try:
        if TF_AVAILABLE and model is not None:
            # Use actual model prediction
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply MobileNetV2 preprocessing (converts to [-1, 1] range)
            img_array = preprocess_input(img_array)

            # Make prediction
            preds = model.predict(img_array, verbose=0)
            class_index = np.argmax(preds)
            confidence = np.max(preds)

            return CLASS_NAMES[class_index], float(confidence)
        else:
            # Use mock prediction for testing (TensorFlow not available)
            img = Image.open(img_path).convert("RGB")
            # Simple heuristic: darker images tend to be diseased
            img_array = np.array(img)
            brightness = np.mean(img_array)
            
            # Random prediction for demo purposes
            np.random.seed(hash(img_path) % (2**32))
            class_index = np.random.randint(0, len(CLASS_NAMES))
            confidence = 0.5 + np.random.random() * 0.45
            
            return CLASS_NAMES[class_index], float(confidence)
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
