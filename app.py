import streamlit as st
from PIL import Image
import os
from utils import predict_image

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image, and the model will classify it as Healthy or Diseased.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # save temporarily
    temp_path = "temp_image.jpg"
    img.save(temp_path)

    # predict
    label, confidence = predict_image(temp_path)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    os.remove(temp_path)
