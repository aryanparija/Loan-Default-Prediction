from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(
    title="Loan Default Risk API",
    description="Predict Loan Default Probability",
    version="1.0"
)

# Load trained pipeline
model = joblib.load("models/loan_default_model.pkl")


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "LightGBM Loan Default Predictor"
    }


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    if probability < 0.10:
        risk_tier = "A"
        decision = "APPROVE"

    elif probability < 0.25:
        risk_tier = "B"
        decision = "APPROVE_WITH_REVIEW"

    elif probability < 0.50:
        risk_tier = "C"
        decision = "MANUAL_REVIEW"

    else:
        risk_tier = "D"
        decision = "REJECT"

    return {
        "default_probability": round(probability, 4),
        "risk_tier": risk_tier,
        "decision": decision
    }