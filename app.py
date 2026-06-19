import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

# Load trained models
model = pickle.load(open("brain_mri_xgb.pkl", "rb"))
pca = pickle.load(open("pca_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# App title
st.title("Brain MRI Abnormality Detection Agent")
st.write("Upload a Brain MRI image for screening.")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])


# Prediction function
def predict_mri(image):
    # Convert uploaded image to RGB
    image = image.convert("RGB")
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (128, 128))

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Flatten
    feature = img.flatten().reshape(1, -1)

    # PCA transform
    feature_pca = pca.transform(feature)

    # Prediction
    pred = model.predict(feature_pca)[0]

    # Prediction probabilities
    probs = model.predict_proba(feature_pca)
    confidence = np.max(probs) * 100

    # Decode label
    label = encoder.inverse_transform([pred])[0]

    return label, confidence


# Main app logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    result, confidence = predict_mri(image)

    st.subheader("Prediction Result")
    st.write("Predicted Class:", result)
    st.write("Confidence:", round(confidence, 2), "%")

    if result == "notumor":
        st.success("Low Risk: No abnormality detected")
    else:
        st.error("High Risk: Abnormality detected")
        st.warning("Consult neurologist immediately.")
