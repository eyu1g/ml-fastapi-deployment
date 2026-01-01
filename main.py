from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ----- FASTAPI APP -----
app = FastAPI()

# ----- CORS CONFIG -----
origins = [
    "https://ml-frontend-c716.vercel.app",  # your Vercel frontend
    "http://localhost:3000",                 # for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- REQUEST BODY MODEL -----
class PredictRequest(BaseModel):
    features: List[float]  # list of numbers from input
    model: str             # selected model name

# ----- MOCK MODELS (replace with real ML model later) -----
def logistic_regression_predict(features):
    # Just a dummy prediction logic
    return 1 if sum(features) > 1000 else 0

def decision_tree_predict(features):
    # Just a dummy prediction logic
    return 1 if features[0] > 50 else 0

# ----- PREDICTION ENDPOINT -----
@app.post("/predict")
def predict(data: PredictRequest):
    if data.model == "Logistic Regression":
        prediction = logistic_regression_predict(data.features)
    elif data.model == "Decision Tree":
        prediction = decision_tree_predict(data.features)
    else:
        return {"error": "Unknown model"}
    
    return {"prediction": prediction}
