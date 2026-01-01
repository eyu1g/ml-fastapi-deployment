from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Heart Disease Prediction API")

# Allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler
logistic_model = joblib.load("logistic_model.joblib")          # Logistic Regression model
decision_tree_model = joblib.load("decision_tree_model.joblib")  # Decision Tree model
scaler = joblib.load("scaler.joblib")                          # Scaler

class PredictRequest(BaseModel):
    model: str             # "Logistic Regression" or "Decision Tree"
    features: str          # comma-separated features as string

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # Normalize and convert features to float array
        features = np.array([float(x.strip()) for x in data.features.split(",")]).reshape(1, -1)
        features = scaler.transform(features)
    except Exception as e:
        return {"error": f"Invalid input: {str(e)}"}

    try:
        # Normalize model name
        model_name = data.model.strip().lower()

        if model_name in ["logistic regression", "logistic"]:
            prediction = logistic_model.predict(features)[0]
        elif model_name in ["decision tree", "tree"]:
            prediction = decision_tree_model.predict(features)[0]
        else:
            return {"error": "Invalid model. Choose 'Logistic Regression' or 'Decision Tree'."}

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
