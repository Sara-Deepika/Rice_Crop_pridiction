
import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Rice Crop Production Prediction",
    page_icon="🌾",
    layout="centered"
)

# Load ML model
MODEL_PATH = Path(__file__).parent / "rice_crop_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

# Application title
st.title("🌾 Rice Crop Production Prediction")

st.write(
    "Predict Paddy Yield using Machine Learning"
)

st.divider()

# Input
hectares = st.number_input(
    "Enter Hectares",
    min_value=0.01,
    value=1.0,
    step=0.1
)

# Prediction
if st.button("Predict Paddy Yield", use_container_width=True):

    try:

        input_data = pd.DataFrame({
            "Hectares": [hectares]
        })

        prediction = model.predict(input_data)

        st.success("Prediction Successful!")

        st.metric(
            "Predicted Paddy Yield",
            f"{float(prediction[0]):,.2f} Kg"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

st.divider()

st.caption(
    "Rice Crop Production Prediction | Machine Learning Project"
)
