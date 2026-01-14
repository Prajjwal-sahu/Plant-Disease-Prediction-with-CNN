import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Get current working directory (app folder)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Paths for model and class labels
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load trained CNN model
model = tf.keras.models.load_model(model_path)

# Load class index mapping
class_indices = json.load(open(class_indices_path))


# Function to load & preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


# Function to predict disease
def predict_image_class(model, image, class_indices):
    processed_img = load_and_preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[str(predicted_index)]
    return predicted_class


# Streamlit UI
st.title("🌿 Plant Disease Prediction System")

uploaded_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((200, 200)), caption="Uploaded Image")

    with col2:
        if st.button("Predict"):
            result = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Predicted Disease: {result}")
