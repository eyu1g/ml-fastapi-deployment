from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Heart Disease Prediction API")

# ✅ CORRECT CORS CONFIG
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ml-frontend-c716.vercel.app",  # ✅ YOUR REAL FRONTEND
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    features: List[float]
    model: str

def logistic_regression_predict(features: List[float]) -> int:
    return 1 if sum(features) > 1000 else 0

def decision_tree_predict(features: List[float]) -> int:
    return 1 if features[0] > 50 else 0

@app.post("/predict")
def predict(data: PredictRequest):
    if data.model == "Logistic Regression":
        prediction = logistic_regression_predict(data.features)
    elif data.model == "Decision Tree":
        prediction = decision_tree_predict(data.features)
    else:
        return {"error": "Unknown model selected"}

    return {"prediction": prediction}
