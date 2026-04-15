# src/predict.py

import cv2
import numpy as np
import joblib

IMG_SIZE = 32

def predict_image(image_path):

    # Load model
    model = joblib.load("model/model.pkl")

    # Read image
    img = cv2.imread(image_path)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Flatten
    img = img.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img)

    return "Cat 🐱" if prediction[0] == 0 else "Dog 🐶"