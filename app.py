import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "saved_models/cnn_with_augmentation.keras"
    )
    return model

model = load_model()

# -----------------------------
# Class names (same order as training)
# -----------------------------
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to predict the disease.")

uploaded_file = st.file_uploader(
    "Choose a rice leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")


    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing the leaf..."):
            predictions = model.predict(img_array)
            confidence = np.max(predictions)
            predicted_class = class_names[np.argmax(predictions)]

        st.success(f"🦠 Predicted Disease: **{predicted_class}**")
        st.write(f"🔍 Confidence: **{confidence * 100:.2f}%**")
