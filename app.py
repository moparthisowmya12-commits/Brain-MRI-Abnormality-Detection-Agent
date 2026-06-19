import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

# Load models
model = pickle.load(open("brain_mri_xgb.pkl", "rb"))
pca = pickle.load(open("pca_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Brain MRI Abnormality Detection Agent")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

def predict_mri(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    feature = img.flatten().reshape(1, -1)
    feature_pca = pca.transform(feature)

    pred = model.predict(feature_pca)[0]
    label = encoder.inverse_transform([pred])[0]

    return label

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    result = predict_mri(image)

    st.subheader("Prediction Result")
    st.success(result)

    if result == "notumor":
        st.success("Low Risk: No abnormality detected")
    else:
        st.error("High Risk: Consult neurologist immediately")
