import os
import sys
import tensorflow as tf
from tensorflow import keras

print('Python version:', sys.version)
print('TensorFlow version:', tf.__version__)
try:
    print('Keras version:', tf.keras.__version__)
except:
    print('Keras version: Unable to determine')

# Try to load the model the same way utils.py does
def get_model_path():
    """Get the path to the model file, preferring .h5 over .keras"""
    model_dir = "model"
    h5_path = os.path.join(model_dir, "plant_disease_model.h5")
    keras_path = os.path.join(model_dir, "plant_disease_model.keras")

    if os.path.exists(h5_path):
        print(f"Using HDF5 model: {h5_path}")
        return h5_path
    elif os.path.exists(keras_path):
        print(f"Using Keras model: {keras_path}")
        return keras_path
    else:
        raise FileNotFoundError("No model file found")

try:
    model_path = get_model_path()
    print(f"Loading model from: {model_path}")

    # Try loading with compile=False first
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")

    # Save as HDF5 if it's not already
    if not model_path.endswith('.h5'):
        h5_path = model_path.replace('.keras', '.h5')
        print(f"Saving as HDF5: {h5_path}")
        model.save(h5_path, save_format='h5')
        print("✅ Saved as HDF5!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()