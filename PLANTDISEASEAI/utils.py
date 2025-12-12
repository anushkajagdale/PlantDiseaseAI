import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "model/plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Change these AFTER you see the folder names inside dataset/train
CLASS_NAMES = sorted(os.listdir("dataset/train"))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    return CLASS_NAMES[class_index], confidence
