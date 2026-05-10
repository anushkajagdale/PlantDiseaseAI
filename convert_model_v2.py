import tensorflow as tf
import os

model_path = 'model/plant_disease_model.keras'
h5_path = 'model/plant_disease_model.h5'

print("Attempting to convert model...")

try:
    # Try loading with safe_mode
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    print("Loaded with safe_mode=False")
except Exception as e:
    print(f"Failed with safe_mode: {e}")
    try:
        # Try loading without compile
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Loaded with compile=False")
    except Exception as e2:
        print(f"Failed with compile=False: {e2}")
        try:
            # Try loading with custom objects
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
            print("Loaded with custom_objects")
        except Exception as e3:
            print(f"All loading methods failed: {e3}")
            exit(1)

# Save as HDF5
model.save(h5_path, save_format='h5')
print(f"Model saved as {h5_path}")

# Verify
if os.path.exists(h5_path):
    print("HDF5 file created successfully")
    size = os.path.getsize(h5_path)
    print(f"File size: {size} bytes")
else:
    print("Failed to create HDF5 file")