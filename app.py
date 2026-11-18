
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Rice Crop Production Prediction",
    page_icon="🌾",
    layout="centered"
)


# --------------------------------
# 2. Load Trained Model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rice_crop_model.pkl"


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    loaded_models = load_model()

    if isinstance(loaded_models, dict):
        rice_models = {
            name: model
            for name, model in loaded_models.items()
            if hasattr(model, "predict")
        }
    else:
        rice_models = {"Default Model": loaded_models}

    if not rice_models:
        raise ValueError("No prediction models found.")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# --------------------------------
# 3. Application Title
# --------------------------------

st.title("🌾 Rice Crop Production Prediction")

st.write(
    "Predict rice crop production using Machine Learning."
)

st.divider()


# --------------------------------
# 4. Select Model
# --------------------------------

model_name = st.selectbox(
    "Select Machine Learning Model",
    list(rice_models.keys())
)

model = rice_models[model_name]


# --------------------------------
# 5. Get Input Features
# --------------------------------

st.subheader("Enter Crop Details")

feature_names = getattr(
    model,
    "feature_names_in_",
    None
)

if feature_names is not None:

    feature_names = list(feature_names)

    user_inputs = {}

    for feature in feature_names:

        user_inputs[feature] = st.number_input(
            f"Enter {feature}",
            value=0.0,
            format="%.4f"
        )

    input_data = pd.DataFrame(
        [user_inputs],
        columns=feature_names
    )

else:

    number_of_features = getattr(
        model,
        "n_features_in_",
        None
    )

    if number_of_features is None:
        st.error(
            "Model feature information is unavailable. "
            "Please check your trained model."
        )
        st.stop()

    user_inputs = []

    for i in range(number_of_features):

        value = st.number_input(
            f"Feature {i + 1}",
            value=0.0
        )

        user_inputs.append(value)

    input_data = np.array([user_inputs])


# --------------------------------
# 6. Prediction Button
# --------------------------------

st.divider()

if st.button(
    "Predict Rice Crop Production",
    type="primary",
    use_container_width=True
):

    try:

        prediction = model.predict(input_data)

        predicted_value = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        st.success("Prediction Completed Successfully!")

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Paddy Yield",
            value=f"{predicted_value:,.2f} kg"
        )

        st.balloons()

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# --------------------------------
# 7. Footer
# --------------------------------

st.divider()

st.caption(
    "Rice Crop Production Prediction | "
    "Machine Learning Project"
)
