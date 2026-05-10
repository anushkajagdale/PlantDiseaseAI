import tensorflow as tf
import os

# Load the existing .keras model
model_path = 'model/plant_disease_model.keras'
h5_path = 'model/plant_disease_model.h5'

if os.path.exists(model_path):
    print("Loading model from .keras format...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Save as HDF5 format
        print(f"Saving model to HDF5 format at {h5_path}...")
        model.save(h5_path, save_format='h5')
        print("Model conversion completed successfully!")

        # Verify the saved model
        if os.path.exists(h5_path):
            print(f"HDF5 model saved at {h5_path}")
        else:
            print("Error: HDF5 model was not saved")

    except Exception as e:
        print(f"Error during model conversion: {e}")
        print("Trying alternative loading method...")

        # Try loading with compile=False
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded with compile=False")
            model.save(h5_path, save_format='h5')
            print("Model conversion completed with alternative method!")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

else:
    print(f"Model file not found at {model_path}")