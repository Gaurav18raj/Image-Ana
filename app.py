# app.py

import streamlit as st
import cv2
import numpy as np
import joblib

IMG_SIZE = 32

# Load model
model = joblib.load("model/model.pkl")

st.title("🐱🐶 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    # Resize & flatten
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img_resized.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_flat)

    result = "Cat 🐱" if prediction[0] == 0 else "Dog 🐶"

    st.success(f"Prediction: {result}")