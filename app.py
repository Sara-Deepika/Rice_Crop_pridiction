
from pathlib import Path
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Rice Crop Production Prediction API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "rice_crop_model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError("rice_crop_model.pkl not found!")

model = joblib.load(MODEL_PATH)


# Input schema
class PredictionInput(BaseModel):
    model_name: str = "simple_model"
    features: dict


# Home route
@app.get("/")
def home():
    return {
        "message": "Rice Crop Prediction API is running"
    }


# Prediction route
@app.post("/predict")
def predict(data: PredictionInput):

    try:
        hectares = data.features.get("Hectares")

        if hectares is None or float(hectares) <= 0:
            raise HTTPException(
                status_code=400,
                detail="Enter valid Hectares"
            )

        input_data = np.array(
            [[float(hectares)]]
        )

        prediction = model.predict(input_data)

        return {
            "predicted_paddy_yield_kg": float(prediction[0])
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
