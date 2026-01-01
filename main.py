from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load models
logistic_model = joblib.load("logistic_model.joblib")
decision_tree_model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    model_type = data["model"]  # "logistic" or "tree"
    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)

    if model_type == "logistic":
        prediction = logistic_model.predict(features)[0]
    else:
        prediction = decision_tree_model.predict(features)[0]

    return {"prediction": int(prediction)}
